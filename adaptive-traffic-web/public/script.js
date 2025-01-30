document.addEventListener("DOMContentLoaded", () => {
    fetch("/api/main")
        .then((response) => response.json())
        .then((data) => {
            const resultsDiv = document.getElementById("results");
            resultsDiv.innerHTML = `
                <p><strong>Traffic Volume:</strong> ${data.data.traffic_volume}</p>
                <p><strong>Vehicle Speed:</strong> ${data.data.vehicle_speed}</p>
                <p><strong>Congestion Level:</strong> ${data.data.congestion_level}</p>
                <p><strong>Vehicle Count:</strong> ${data.data.vehicle_count}</p>
                <p><strong>Predicted Congestion:</strong> ${data.predictions[0]}</p>
                <p><strong>Signal Timing:</strong> ${data.signal_timings[0]} seconds</p>
                <p><strong>Recommended Route:</strong> ${data.routes[0]}</p>
            `;
        })
        .catch((error) => {
            console.error("Error fetching data:", error);
        });
});