import os
import pika
import json
from dotenv import load_dotenv

load_dotenv()

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")


class RabbitMQPublisher:
    def __init__(self, host: str = None, queue_name: str = "ocr_task_queue"):
        self.host = host or RABBITMQ_HOST
        self.queue_name = queue_name

    def _get_channel(self):
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))
            channel = connection.channel()
            channel.queue_declare(queue=self.queue_name, durable=True)
            return connection, channel
        except Exception as e:
            print(f"[BrokerService] Failed to connect to RabbitMQ: {e}")
            return None, None

    def publish_task(self, pdf_name: str) -> bool:
        connection, channel = self._get_channel()
        if not channel:
            return False
            
        task_payload = json.dumps({"pdf_name": pdf_name})
        
        try:
            channel.basic_publish(
                exchange='',
                routing_key=self.queue_name,
                body=task_payload,
                properties=pika.BasicProperties(
                    delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
                )
            )
            print(f"[BrokerService] Published Task: {pdf_name}")
            return True
        except Exception as e:
            print(f"[BrokerService] Failed to publish message: {e}")
            return False
        finally:
            if connection and not connection.is_closed:
                connection.close()

# Provide singleton instances for dependency injection
ocr_publisher = RabbitMQPublisher(queue_name="ocr_task_queue")
to_md_publisher = RabbitMQPublisher(queue_name="to_md_task_queue")
cleaning_publisher = RabbitMQPublisher(queue_name="cleaning_task_queue")
chunking_publisher = RabbitMQPublisher(queue_name="chunking_task_queue")
embedding_publisher = RabbitMQPublisher(queue_name="embedding_task_queue")
