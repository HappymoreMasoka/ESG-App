import os
from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter


def setup_telemetry() -> None:
    service_name = os.getenv("OTEL_SERVICE_NAME", "esg-risk-dashboard")
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    resource = Resource.create({
        "service.name": service_name,
        "service.version": "2.0.0",
        "deployment.environment": os.getenv("APP_ENV", "development"),
    })

    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    metric_exporter = OTLPMetricExporter(endpoint=f"{otlp_endpoint}/v1/metrics")
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)


tracer = trace.get_tracer("esg-risk-dashboard")
meter = metrics.get_meter("esg-risk-dashboard")

prediction_counter = meter.create_counter(
    name="esg_predictions_total",
    description="Total number of ESG prediction requests"
)

prediction_latency = meter.create_histogram(
    name="esg_prediction_latency_ms",
    description="Prediction latency in milliseconds"
)

nlp_counter = meter.create_counter(
    name="esg_nlp_requests_total",
    description="Total number of NLP analysis requests"
)