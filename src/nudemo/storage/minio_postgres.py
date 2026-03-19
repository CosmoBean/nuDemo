from __future__ import annotations

import io
import time
from contextlib import closing
from dataclasses import dataclass

from nudemo.config import MinioSettings, PostgresSettings
from nudemo.storage.base import StorageWriteResult, array_to_npy_bytes, image_to_jpeg_bytes


@dataclass(slots=True)
class MinioPostgresBackend:
    minio: MinioSettings
    postgres: PostgresSettings
    name: str = "MinIO+PostgreSQL"

    def _clients(self):
        import psycopg
        from minio import Minio

        minio_client = Minio(
            self.minio.endpoint,
            access_key=self.minio.access_key,
            secret_key=self.minio.secret_key,
            secure=self.minio.secure,
        )
        connection = psycopg.connect(self.postgres.dsn)
        return minio_client, connection

    def write_samples(self, samples):
        minio_client, connection = self._clients()
        t0 = time.perf_counter()
        bytes_written = 0
        if not minio_client.bucket_exists(self.minio.bucket):
            minio_client.make_bucket(self.minio.bucket)

        with connection, closing(connection.cursor()) as cursor:
            cursor.execute("TRUNCATE annotations, samples, scenes RESTART IDENTITY CASCADE")
            samples_written = 0
            for sample_idx, sample in enumerate(samples):
                cursor.execute(
                    """
                    INSERT INTO scenes (scene_token, scene_name, location, num_samples)
                    VALUES (%s, %s, %s, 1)
                    ON CONFLICT (scene_token)
                    DO UPDATE SET num_samples = scenes.num_samples + 1
                    """,
                    (
                        sample.scene_token,
                        sample.scene_name,
                        sample.location,
                    ),
                )

                blob_refs = sample.blob_refs(sample_idx)
                for camera, path in blob_refs.camera_paths.items():
                    payload = image_to_jpeg_bytes(sample.cameras[camera])
                    bytes_written += len(payload)
                    minio_client.put_object(
                        self.minio.bucket,
                        path,
                        data=io.BytesIO(payload),
                        length=len(payload),
                        content_type="image/jpeg",
                    )

                for sensor, path in blob_refs.sensor_paths.items():
                    data = sample.lidar_top if sensor == "LIDAR_TOP" else sample.radars[sensor]
                    payload = array_to_npy_bytes(data)
                    bytes_written += len(payload)
                    minio_client.put_object(
                        self.minio.bucket,
                        path,
                        data=io.BytesIO(payload),
                        length=len(payload),
                        content_type="application/octet-stream",
                    )

                flat_refs = blob_refs.flattened()
                cursor.execute(
                    """
                    INSERT INTO samples (
                        sample_idx, token, scene_token, timestamp, location,
                        ego_translation, ego_rotation, num_annotations, num_lidar_points,
                        cam_front_path, cam_front_left_path, cam_front_right_path,
                        cam_back_path, cam_back_left_path, cam_back_right_path,
                        lidar_top_path, radar_front_path, radar_front_left_path,
                        radar_front_right_path, radar_back_left_path, radar_back_right_path
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (sample_idx) DO UPDATE SET token = EXCLUDED.token
                    """,
                    (
                        sample_idx,
                        sample.token,
                        sample.scene_token,
                        sample.timestamp,
                        sample.location,
                        sample.ego_translation,
                        sample.ego_rotation,
                        len(sample.annotations),
                        int(sample.lidar_top.shape[0]),
                        flat_refs["CAM_FRONT"],
                        flat_refs["CAM_FRONT_LEFT"],
                        flat_refs["CAM_FRONT_RIGHT"],
                        flat_refs["CAM_BACK"],
                        flat_refs["CAM_BACK_LEFT"],
                        flat_refs["CAM_BACK_RIGHT"],
                        flat_refs["LIDAR_TOP"],
                        flat_refs["RADAR_FRONT"],
                        flat_refs["RADAR_FRONT_LEFT"],
                        flat_refs["RADAR_FRONT_RIGHT"],
                        flat_refs["RADAR_BACK_LEFT"],
                        flat_refs["RADAR_BACK_RIGHT"],
                    ),
                )

                for annotation in sample.annotations:
                    cursor.execute(
                        """
                        INSERT INTO annotations (
                            sample_idx, category, translation, size, rotation,
                            num_lidar_pts, num_radar_pts
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            sample_idx,
                            annotation.category,
                            annotation.translation,
                            annotation.size,
                            annotation.rotation,
                            annotation.num_lidar_pts,
                            annotation.num_radar_pts,
                        ),
                    )
                samples_written += 1

        elapsed = time.perf_counter() - t0
        return StorageWriteResult(
            backend=self.name,
            samples_written=samples_written,
            elapsed_sec=elapsed,
            bytes_written=bytes_written,
        )

    def sequential_iter(self):
        import psycopg
        from minio import Minio

        minio_client = Minio(
            self.minio.endpoint,
            access_key=self.minio.access_key,
            secret_key=self.minio.secret_key,
            secure=self.minio.secure,
        )
        with psycopg.connect(self.postgres.dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT sample_idx, cam_front_path, lidar_top_path FROM samples ORDER BY sample_idx"
            )
            for sample_idx, cam_path, lidar_path in cursor:
                cam_response = minio_client.get_object(self.minio.bucket, cam_path)
                lidar_response = minio_client.get_object(self.minio.bucket, lidar_path)
                yield {
                    "idx": sample_idx,
                    "cam": cam_response.read(),
                    "lidar": lidar_response.read(),
                }
                cam_response.close()
                lidar_response.close()

    def fetch(self, sample_idx: int):
        import psycopg
        from minio import Minio

        minio_client = Minio(
            self.minio.endpoint,
            access_key=self.minio.access_key,
            secret_key=self.minio.secret_key,
            secure=self.minio.secure,
        )
        with psycopg.connect(self.postgres.dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT cam_front_path, lidar_top_path FROM samples WHERE sample_idx = %s",
                (sample_idx,),
            )
            cam_path, lidar_path = cursor.fetchone()
            cam_response = minio_client.get_object(self.minio.bucket, cam_path)
            lidar_response = minio_client.get_object(self.minio.bucket, lidar_path)
            result = {"cam": cam_response.read(), "lidar": lidar_response.read()}
            cam_response.close()
            lidar_response.close()
            return result

    def curation_query(self):
        import psycopg

        with psycopg.connect(self.postgres.dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT s.sample_idx
                FROM samples s
                JOIN annotations a ON a.sample_idx = s.sample_idx
                WHERE s.location = 'boston-seaport'
                  AND a.category LIKE 'human.pedestrian%%'
                  AND s.num_annotations > 5
                ORDER BY s.sample_idx
                """
            )
            return [row[0] for row in cursor.fetchall()]

    def disk_footprint(self) -> int:
        return 0
