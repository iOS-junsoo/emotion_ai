from typing import Optional, Dict, Any
import os

try:
    from kafka import KafkaProducer  # type: ignore
except Exception:  # pragma: no cover
    KafkaProducer = None  # type: ignore


class KafkaPub:
    """
    Kafka 퍼블리시(선택) 스텁.
    """

    def __init__(self, bootstrap: Optional[str] = None, topic: Optional[str] = None) -> None:
        self.bootstrap = bootstrap or os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
        self.topic = topic or os.getenv("KAFKA_TOPIC", "emotion_events")
        self.producer = KafkaProducer(bootstrap_servers=self.bootstrap) if KafkaProducer else None

    def publish(self, payload: Dict[str, Any]) -> None:
        if not self.producer:
            return
        self.producer.send(self.topic, value=str(payload).encode("utf-8"))
        self.producer.flush()


