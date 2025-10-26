// 🚨 Emergency Alert System for Medical AI
// Displays urgent popup when dangerous symptoms detected

function showEmergencyAlert(message) {
    // Extract emergency type from message
    let emergencyType = 'MEDICAL EMERGENCY';
    if (message.includes('HEART ATTACK')) emergencyType = '⚠️ HEART ATTACK';
    else if (message.includes('STROKE')) emergencyType = '⚠️ STROKE';  
    else if (message.includes('ALLERGIC REACTION')) emergencyType = '⚠️ SEVERE ALLERGIC REACTION';
    else if (message.includes('MENINGITIS')) emergencyType = '⚠️ MENINGITIS';
    else if (message.includes('BLEEDING')) emergencyType = '⚠️ SEVERE BLEEDING';
    
    // Create custom alert modal
    const alertHTML = `
        <div id="emergencyAlert" style="
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 10000;
            display: flex;
            justify-content: center;
            align-items: center;
            animation: fadeIn 0.3s;
        ">
            <div style="
                background: linear-gradient(135deg, #ff0000, #dc143c);
                color: white;
                padding: 30px;
                border-radius: 15px;
                max-width: 600px;
                box-shadow: 0 10px 50px rgba(255, 0, 0, 0.7);
                animation: slideIn 0.5s;
                border: 4px solid white;
            ">
                <div style="text-align: center; margin-bottom: 20px;">
                    <div style="font-size: 70px; animation: pulse 1s infinite;">🚨</div>
                    <h2 style="margin: 10px 0; font-size: 28px; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
                        ${emergencyType}
                    </h2>
                </div>
                <div style="
                    background: rgba(255, 255, 255, 0.98);
                    color: #000;
                    padding: 25px;
                    border-radius: 10px;
                    font-size: 16px;
                    line-height: 1.8;
                    max-height: 450px;
                    overflow-y: auto;
                    font-weight: 500;
                ">
                    ${message.replace(/\n/g, '<br>').replace(/\*\*/g, '')}
                </div>
                <div style="text-align: center; margin-top: 25px;">
                    <button onclick="closeEmergencyAlert()" style="
                        background: white;
                        color: #dc143c;
                        border: none;
                        padding: 18px 50px;
                        font-size: 20px;
                        font-weight: bold;
                        border-radius: 30px;
                        cursor: pointer;
                        box-shadow: 0 6px 15px rgba(0,0,0,0.3);
                        transition: transform 0.2s;
                    " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                        ✓ I UNDERSTAND
                    </button>
                </div>
                <div style="text-align: center; margin-top: 20px; font-size: 22px;">
                    <a href="tel:911" style="
                        color: white;
                        text-decoration: none;
                        background: rgba(0,0,0,0.4);
                        padding: 12px 30px;
                        border-radius: 25px;
                        display: inline-block;
                        font-weight: bold;
                        transition: background 0.3s;
                    " onmouseover="this.style.background='rgba(0,0,0,0.6)'" onmouseout="this.style.background='rgba(0,0,0,0.4)'">
                        📞 Call Emergency: 911 / 1669
                    </a>
                </div>
            </div>
        </div>
    `;
    
    // Add CSS animations
    if (!document.getElementById('emergencyAlertStyles')) {
        const style = document.createElement('style');
        style.id = 'emergencyAlertStyles';
        style.textContent = `
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            @keyframes slideIn {
                from { transform: translateY(-100px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.3); }
            }
        `;
        document.head.appendChild(style);
    }
    
    // Add alert to page
    document.body.insertAdjacentHTML('beforeend', alertHTML);
    
    // Play alert sound (3 beeps)
    try {
        const beep = () => {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            osc.type = 'sine';
            osc.frequency.value = 880; // High pitch alert
            osc.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.15);
        };
        beep();
        setTimeout(beep, 250);
        setTimeout(beep, 500);
    } catch (e) {
        console.log('⚠️ Audio not supported');
    }
    
    console.log('🚨 EMERGENCY ALERT DISPLAYED:', emergencyType);
}

function closeEmergencyAlert() {
    const alert = document.getElementById('emergencyAlert');
    if (alert) {
        alert.style.animation = 'fadeOut 0.3s';
        setTimeout(() => alert.remove(), 300);
    }
}

// Add fadeOut animation
const fadeOutStyle = document.createElement('style');
fadeOutStyle.textContent = `
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
`;
document.head.appendChild(fadeOutStyle);
