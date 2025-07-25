// Login form handler
document.getElementById('loginForm')?.addEventListener('submit', function(e) {
  e.preventDefault();

  const username = document.getElementById('username').value.trim();
  const password = document.getElementById('password').value.trim();

  if (username && password) {
    window.location.href = "area.html";
  } else {
    alert("Please enter valid login details.");
  }
});

// Predict Crime form handler
const predictForm = document.getElementById('predictCrimeForm');
if (predictForm) {
  predictForm.addEventListener('submit', async function(e) {
    e.preventDefault();

    const area = document.getElementById('area').value.trim();
    const resultDiv = document.getElementById('predictionResult');
    resultDiv.textContent = '🔍 Predicting...';

    try {
      const response = await fetch('http://127.0.0.1:5000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ area })
      });

      const data = await response.json();

      if (data.result) {
        const result = data.result;
        let output = `📍 ${result.message}`;

        if (result.cluster !== null)
          output += ` | Cluster: ${result.cluster}`;
        if (result.avg_crime_weight !== null)
          output += ` | Avg Crime Weight: ${result.avg_crime_weight}`;

        if (result.crimes && result.crimes.length > 0) {
          output += `\n\n🕵️‍♀️ Crimes in this area:\n`;
          result.crimes.forEach((crime, i) => {
            output += `${i + 1}. Year: ${crime.Year}, Type: ${crime.Type}, Weight: ${crime.Crime_Weight}\n`;
          });
        }

        resultDiv.textContent = output;

        // Redirect to dashboard
        if (result.cluster !== null) {
          setTimeout(() => {
            window.location.href = 'crime_dashboard'; // ✅ Adjust as needed
          }, 2000);
        }

      } else {
        resultDiv.textContent = data.error || 'No prediction result.';
      }

    } catch (err) {
      console.error(err);
      resultDiv.textContent = '❌ Error connecting to backend.';
    }
  });
}
