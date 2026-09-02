import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


URL = "http://127.0.0.1:8000/api/v1/predict"

TOTAL_REQUESTS = 100
CONCURRENT_USERS = 20


def send_request(request_number: int) -> dict:
    payload = {
        "text": f"My package has not arrived {request_number}"
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        URL,
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    start_time = time.perf_counter()

    try:
        with urllib.request.urlopen(
            request,
            timeout=30,
        ) as response:

            body = response.read().decode("utf-8")
            status_code = response.status

        duration = time.perf_counter() - start_time

        return {
            "success": True,
            "duration": duration,
            "status_code": status_code,
            "body": body,
        }

    except Exception as exc:
        duration = time.perf_counter() - start_time

        return {
            "success": False,
            "duration": duration,
            "error": str(exc),
        }


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:

    if not values:
        return 0.0

    sorted_values = sorted(values)

    index = math.ceil(
        percentile_value * len(sorted_values)
    ) - 1

    return sorted_values[index]


def main():
    start_time = time.perf_counter()

    results = []

    with ThreadPoolExecutor(
        max_workers=CONCURRENT_USERS
    ) as executor:

        futures = [
            executor.submit(
                send_request,
                i,
            )
            for i in range(TOTAL_REQUESTS)
        ]

        for future in as_completed(futures):
            results.append(
                future.result()
            )

    total_time = time.perf_counter() - start_time

    successful = [
        result
        for result in results
        if result["success"]
    ]

    failed = [
        result
        for result in results
        if not result["success"]
    ]

    latencies = [
        result["duration"]
        for result in successful
    ]

    throughput = (
        len(successful) / total_time
        if total_time > 0
        else 0
    )

    p50 = percentile(
        latencies,
        0.50,
    )

    p95 = percentile(
        latencies,
        0.95,
    )

    p99 = percentile(
        latencies,
        0.99,
    )

    print()
    print("Load Test Results")
    print("-----------------")

    print(
        f"Total requests: {TOTAL_REQUESTS}"
    )

    print(
        f"Concurrent users: {CONCURRENT_USERS}"
    )

    print(
        f"Successful: {len(successful)}"
    )

    print(
        f"Failed: {len(failed)}"
    )

    print(
        f"Total time: {total_time:.2f} seconds"
    )

    print(
        f"Throughput: {throughput:.2f} requests/second"
    )

    if latencies:
        print(
            f"Average latency: "
            f"{sum(latencies) / len(latencies):.3f} seconds"
        )

        print(
            f"P50 latency: {p50:.3f} seconds"
        )

        print(
            f"P95 latency: {p95:.3f} seconds"
        )

        print(
            f"P99 latency: {p99:.3f} seconds"
        )

        print(
            f"Fastest request: "
            f"{min(latencies):.3f} seconds"
        )

        print(
            f"Slowest request: "
            f"{max(latencies):.3f} seconds"
        )

    if failed:
        print()
        print("Failed requests:")

        for result in failed[:5]:
            print(
                result["error"]
            )


if __name__ == "__main__":
    main()