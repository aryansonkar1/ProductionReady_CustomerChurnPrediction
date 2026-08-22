document.getElementById('churn-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const predictBtn = document.getElementById('predict-btn');
    predictBtn.innerText = 'Analyzing...';
    predictBtn.disabled = true;

    // Grab form values
    const tenure = parseInt(document.getElementById('tenure').value);
    const monthly_charges = parseFloat(document.getElementById('monthly-charges').value);
    const contract = document.getElementById('contract').value;

    const payload = {
        tenure: tenure,
        monthly_charges: monthly_charges,
        contract: contract
    };

    try {
        // You may need to change 8002 to wherever the server is currently mapped in dev
        const response = await fetch('http://127.0.0.1:8002/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error('API Error');
        }

        const data = await response.json();
        
        displayResult(data.churn_probability, data.threshold);

    } catch (error) {
        console.error(error);
        alert('Failed to connect to the Churn API. Is the server running?');
    } finally {
        predictBtn.innerText = 'Analyze Risk';
        predictBtn.disabled = false;
    }
});

function displayResult(probability, threshold) {
    const overlay = document.getElementById('result-overlay');
    const ringProgress = document.getElementById('prob-ring');
    const probText = document.getElementById('prob-text');
    const riskStatus = document.getElementById('risk-status');

    // UI resets
    overlay.classList.remove('hidden');
    
    // Slight delay to allow display flex to apply before opacity transition
    setTimeout(() => {
        overlay.classList.add('active');
        
        // Calculate circle properties
        const percentage = Math.round(probability * 100);
        const radius = 40;
        const circumference = 2 * Math.PI * radius; // 251.2
        const offset = circumference - (percentage / 100) * circumference;
        
        ringProgress.style.strokeDashoffset = offset;

        // Counter animation
        animateCounter(probText, percentage);

        // Styling based on risk threshold
        if (probability >= threshold) {
            ringProgress.className.baseVal = "ring-progress high-risk";
            riskStatus.innerText = "High Churn Risk";
            riskStatus.className = "high-risk-text";
        } else {
            ringProgress.className.baseVal = "ring-progress low-risk";
            riskStatus.innerText = "Low Churn Risk";
            riskStatus.className = "low-risk-text";
        }
    }, 50);
}

document.getElementById('close-btn').addEventListener('click', () => {
    const overlay = document.getElementById('result-overlay');
    overlay.classList.remove('active');
    
    setTimeout(() => {
        overlay.classList.add('hidden');
        document.getElementById('prob-ring').style.strokeDashoffset = 251.2;
        document.getElementById('prob-text').innerText = "0%";
    }, 300);
});

// A small counter animation for the circle
function animateCounter(element, target) {
    let current = 0;
    const increment = target / 20; 
    
    const interval = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.innerText = target + '%';
            clearInterval(interval);
        } else {
            element.innerText = Math.round(current) + '%';
        }
    }, 40);
}
