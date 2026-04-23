"""
Cleaning Worker – Lắng nghe hàng đợi `cleaning_task_queue`.
Nhận tên file .md, chạy Cleaning Engine, rồi đẩy tiếp sang Chunking Queue.

Pipeline:
  OCR Worker / ToMD Worker
       ↓ publish md_file_name
  cleaning_task_queue   ← Worker này
       ↓ publish md_file_name (sau khi làm sạch)
  chunking_task_queue
"""
import os
import sys
import pika
import json
from dotenv import load_dotenv

load_dotenv()

# Clean imports
from pipeline.engines.cleaning_engine import run_cleaning_pipeline
from app.services.broker_service import chunking_publisher

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
QUEUE_NAME = "cleaning_task_queue"


def callback(ch, method, properties, body):
    task_data = json.loads(body)
    md_file_name = task_data.get("pdf_name")  # Key chuẩn của hệ thống

    if not md_file_name:
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    print(f"\n[Cleaning Worker] Nhận file: {md_file_name}")

    success = run_cleaning_pipeline(md_file_name)

    if success:
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"[Cleaning Worker] Làm sạch xong. Đẩy sang Chunking: {md_file_name}")
        chunking_publisher.publish_task(md_file_name)
    else:
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        print(f"[Cleaning Worker] Thất bại, loại bỏ task: {md_file_name}")


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        heartbeat=0
    ))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)

    print('========================================================')
    print(' [*] Cleaning Worker ONLINE – Đang chờ file .md cần làm sạch...')
    print('========================================================')
    channel.start_consuming()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[Worker] Đã ngưng hoạt động.')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
