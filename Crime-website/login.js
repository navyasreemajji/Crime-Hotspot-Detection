async function checkCrime() {
  const response = await fetch("http://127.0.0.1:5000/api/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      area: document.getElementById("area").value,
      time: document.getElementById("time").value
    })
  });

  const data = await response.json();
  alert("Prediction: " + data.result);
}
