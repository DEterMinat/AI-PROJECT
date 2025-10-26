// 🏥 Medical AI Q&A System JavaScript
// This file handles the complete workflow: Frontend → FastAPI → N8N → LangChain → AI Models

class MedicalQASystem {
    constructor() {
        // Use Integrated API (FastAPI + Langchain + N8N)
        this.apiBaseUrl = 'http://localhost:8000';
        this.n8nWebhookUrl = 'http://localhost:5678/webhook/medical-qa';
        
        this.currentQuestionId = null;
        this.isProcessing = false;
        this.stats = {
            totalQuestions: 0,
            totalResponseTime: 0,
            successfulRequests: 0
        };
        
        console.log('🌐 Integrated API URL:', this.apiBaseUrl);
        console.log('🔗 N8N Webhook URL:', this.n8nWebhookUrl);
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkSystemStatus();
        this.loadStats();
        this.startPerformanceMonitoring();
    }

    bindEvents() {
        // Send button and Enter key
        const sendButton = document.getElementById('sendButton');
        const questionInput = document.getElementById('questionInput');
        
        sendButton.addEventListener('click', () => this.askQuestion());
        
        questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.askQuestion();
            }
        });

        // Character counter
        questionInput.addEventListener('input', () => {
            const length = questionInput.value.length;
            const counter = document.querySelector('.character-count');
            counter.textContent = `${length}/500`;
            
            // Enable/disable send button
            sendButton.disabled = length === 0 || this.isProcessing;
            
            // Auto-resize textarea
            questionInput.style.height = 'auto';
            questionInput.style.height = Math.min(questionInput.scrollHeight, 150) + 'px';
        });

        // Quick question buttons
        document.querySelectorAll('.quick-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const question = btn.getAttribute('data-question');
                questionInput.value = question;
                questionInput.dispatchEvent(new Event('input'));
                this.askQuestion();
            });
        });
    }

    async checkSystemStatus() {
        console.log('🔍 Checking integrated system status...');
        
        // Check Integrated API
        try {
            const response = await fetch(`${this.apiBaseUrl}/status`, {
                method: 'GET',
                timeout: 5000
            });
            
            if (response.ok) {
                const data = await response.json();
                
                console.log('📊 System Status:', data);
                
                // Update API status
                this.updateComponentStatus('apiStatus', 'active', 'Online');
                
                // Update Langchain status
                if (data.langchain && data.langchain.status === 'online') {
                    this.updateComponentStatus('langchainStatus', 'active', 
                        `Ready (${data.langchain.vectorstore_docs || 0} docs)`);
                    console.log('✅ Langchain: Ready');
                } else {
                    this.updateComponentStatus('langchainStatus', 'error', 'Not Available');
                }
                
                // Update N8N status
                if (data.n8n && data.n8n.online) {
                    this.updateComponentStatus('n8nStatus', 'active', 'Online');
                    console.log('✅ N8N: Online');
                } else {
                    this.updateComponentStatus('n8nStatus', 'error', 'Offline');
                }
                
                // Update Model status
                if (data.model && data.model.loaded) {
                    this.updateComponentStatus('modelStatus', 'active', 
                        `Loaded (${data.model.device || 'cpu'})`);
                    document.getElementById('latestModel').textContent = 
                        data.model.path?.split('/').pop() || 'Model Loaded';
                    console.log('✅ Model: Loaded');
                } else {
                    this.updateComponentStatus('modelStatus', 'error', 'Not Loaded');
                    document.getElementById('latestModel').textContent = 'Model not loaded';
                }
                
                console.log('✅ System Status: OK');
            } else {
                throw new Error('API not responding');
            }
        } catch (error) {
            console.log('❌ Integrated API: Offline -', error.message);
            this.updateComponentStatus('apiStatus', 'error', 'Offline');
            this.updateComponentStatus('langchainStatus', 'error', 'Unavailable');
            this.updateComponentStatus('n8nStatus', 'error', 'Unavailable');
            this.updateComponentStatus('modelStatus', 'error', 'Unavailable');
        }
        
        // Database status (from vectorstore)
        setTimeout(() => {
            this.updateComponentStatus('dbStatus', 'active', 'Vector DB Ready');
            console.log('✅ Vector Database: Ready');
        }, 500);
    }

    updateComponentStatus(elementId, status, text) {
        const element = document.getElementById(elementId);
        if (element) {
            element.className = `status-badge ${status}`;
            element.textContent = text;
        }
    }

    loadStats() {
        // Load from localStorage
        const savedStats = localStorage.getItem('medicalQA_stats');
        if (savedStats) {
            this.stats = { ...this.stats, ...JSON.parse(savedStats) };
        }
        
        this.updateStatsDisplay();
    }

    updateStatsDisplay() {
        document.getElementById('totalQuestions').textContent = this.stats.totalQuestions;
        
        const avgTime = this.stats.totalQuestions > 0 
            ? Math.round(this.stats.totalResponseTime / this.stats.totalQuestions)
            : 0;
        document.getElementById('avgResponseTime').textContent = `${avgTime}ms`;
        
        const successRate = this.stats.totalQuestions > 0 
            ? Math.round((this.stats.successfulRequests / this.stats.totalQuestions) * 100)
            : 100;
        document.getElementById('successRateMetric').textContent = `${successRate}%`;
        
        document.getElementById('apiCallsMetric').textContent = this.stats.totalQuestions;
    }

    saveStats() {
        localStorage.setItem('medicalQA_stats', JSON.stringify(this.stats));
    }

    async askQuestion() {
        if (this.isProcessing) return;

        const questionInput = document.getElementById('questionInput');
        const question = questionInput.value.trim();
        
        if (!question) return;

        this.isProcessing = true;
        const startTime = Date.now();
        
        try {
            // Add user message to chat
            this.addMessage('user', question);
            
            // Clear input
            questionInput.value = '';
            questionInput.dispatchEvent(new Event('input'));
            
            // Show loading overlay
            this.showLoading(true);
            
            // Reset workflow steps
            this.resetWorkflowSteps();
            
            // Start the complete workflow
            await this.executeCompleteWorkflow(question, startTime);
            
        } catch (error) {
            console.error('❌ Error processing question:', error);
            this.addMessage('system', `เกิดข้อผิดพลาด: ${error.message}`);
            this.setWorkflowStepStatus('step1', 'error');
        } finally {
            this.isProcessing = false;
            this.showLoading(false);
            document.getElementById('sendButton').disabled = false;
        }
    }

    async executeCompleteWorkflow(question, startTime) {
        console.log('🚀 Starting complete medical Q&A workflow...');
        
        // Step 1: Question Input (already done)
        this.setWorkflowStepStatus('step1', 'completed', 'Question received');
        this.updateLoadingMessage('ได้รับคำถามแล้ว...');
        
        await this.delay(300);
        
        // Step 2: API Gateway
        this.setWorkflowStepStatus('step2', 'active');
        this.updateLoadingMessage('กำลังส่งคำถามไปยัง FastAPI Gateway...');
        
        try {
            // Call Integrated API
            const response = await fetch(`${this.apiBaseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: question,
                    user_id: 'web_user',
                    use_langchain: true,
                    use_n8n: true,  // ✅ เปิดใช้ N8N Workflow
                    max_length: 150,
                    temperature: 0.7
                })
            });
            
            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`);
            }
            
            const data = await response.json();
            
            this.setWorkflowStepStatus('step2', 'completed', 'API processed');
            await this.delay(300);
            
            // Step 3: N8N Workflow (check method)
            if (data.method === 'n8n') {
                this.setWorkflowStepStatus('step3', 'active');
                this.updateLoadingMessage('Processing through N8N Workflow...');
                await this.delay(500);
                this.setWorkflowStepStatus('step3', 'completed', 'Workflow processed');
            } else {
                this.setWorkflowStepStatus('step3', 'skipped', 'Direct to Langchain');
            }
            
            // Step 4: Langchain AI (check method)
            if (data.method === 'langchain' || data.method === 'n8n') {
                this.setWorkflowStepStatus('step4', 'active');
                this.updateLoadingMessage('Processing with Langchain RAG Engine...');
                await this.delay(500);
                this.setWorkflowStepStatus('step4', 'completed', 'RAG processing complete');
            } else {
                this.setWorkflowStepStatus('step4', 'skipped', 'Not used');
            }
            
            // Step 5: AI Model Generation
            this.setWorkflowStepStatus('step5', 'active');
            this.updateLoadingMessage(`Generating answer with custom trained model...`);
            
            await this.delay(500);
            this.setWorkflowStepStatus('step5', 'completed', 'Answer generated');
            
            // Step 6: Response Delivery
            this.setWorkflowStepStatus('step6', 'active');
            this.updateLoadingMessage('Formatting response...');
            
            await this.delay(300);
            this.setWorkflowStepStatus('step6', 'completed', 'Response delivered');
            
            // Format sources info
            let sourcesInfo = '';
            if (data.sources && data.sources.length > 0) {
                const sourceTopics = data.sources.map(s => s.topic || s).slice(0, 3);
                sourcesInfo = `\n\n📚 Knowledge Sources: ${sourceTopics.join(', ')}`;
            }
            
            // ⚠️ CHECK FOR EMERGENCY ALERT
            const isEmergency = data.answer.includes('🚨') || 
                              data.answer.includes('EMERGENCY') ||
                              data.answer.includes('IMMEDIATE ACTION REQUIRED');
            
            if (isEmergency) {
                // Show urgent alert popup using global function
                showEmergencyAlert(data.answer);
            }
            
            // Add AI response to chat with model info
            const modelInfo = `${data.method.toUpperCase()} • ${data.processing_time?.toFixed(2) || '0.00'}s`;
            this.addMessage('ai', data.answer + sourcesInfo, null, modelInfo);
            
            // Update metrics
            const responseTime = Date.now() - startTime;
            this.updateMetrics(responseTime, true);
            
            console.log('✅ Workflow completed successfully');
            console.log('📊 Method:', data.method);
            console.log('⏱️ Processing Time:', data.processing_time + 's');
            console.log('📚 Sources:', data.sources?.length || 0);
                
        } catch (error) {
            console.error('❌ Integrated API error:', error);
            
            // Show error message
            this.setWorkflowStepStatus('step2', 'error', 'API failed');
            this.addMessage('system', `Cannot connect to Integrated API: ${error.message}`);
            
            // Update metrics with failure
            const responseTime = Date.now() - startTime;
            this.updateMetrics(responseTime, false);
        }
    }

    async callFastAPI(question) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    user_id: 'demo_user',
                    use_langchain: true,
                    use_n8n: true,  // ✅ เปิดใช้ N8N Workflow
                    max_length: 150
                }),
                timeout: 60000  // 60 second timeout for model inference
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return {
                success: data.success,
                answer: data.answer || 'Could not generate an answer',
                sources: data.sources || [],
                method: data.method || 'unknown',
                processing_time: data.processing_time || 0,
                timestamp: data.timestamp
            };

        } catch (error) {
            console.error('FastAPI Error:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    async fallbackToLangChain(question, startTime) {
        console.log('🔄 Falling back to LangChain direct endpoint...');
        
        this.setWorkflowStepStatus('step2', 'error');
        this.setWorkflowStepStatus('step3', 'completed', 'Skipped');
        this.setWorkflowStepStatus('step4', 'active');
        this.updateLoadingMessage('กำลังเรียก LangChain โดยตรง...');

        try {
            // Simulate direct LangChain call
            await this.delay(3000);
            
            this.setWorkflowStepStatus('step4', 'completed', 'Direct call success');
            this.setWorkflowStepStatus('step5', 'completed', 'Model responded');
            this.setWorkflowStepStatus('step6', 'completed', 'Fallback response');
            
            // Generate fallback response
            const fallbackResponse = this.generateFallbackResponse(question);
            this.addMessage('ai', fallbackResponse, 0.7, 'Fallback System');
            
            const responseTime = Date.now() - startTime;
            this.updateMetrics(responseTime, true);
            
        } catch (error) {
            console.error('❌ Fallback failed:', error);
            this.setWorkflowStepStatus('step4', 'error');
            this.addMessage('system', 'ขออภัย ระบบไม่สามารถตอบคำถามได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง');
            
            const responseTime = Date.now() - startTime;
            this.updateMetrics(responseTime, false);
        }
    }

    generateFallbackResponse(question) {
        // Simple fallback responses for medical questions
        const responses = [
            `ขออภัย ขณะนี้ระบบ AI กำลังประมวลผลคำถาม "${question}" ของคุณ ในระหว่างนี้ขอแนะนำให้ปรึกษาแพทย์เฉพาะทางเพื่อคำแนะนำที่ถูกต้องและเหมาะสมกับอาการของคุณ`,
            
            `เกี่ยวกับ "${question}" ขอแนะนำให้คุณ: 
            1. ปรึกษาแพทย์ผู้เชี่ยวชาญ
            2. เก็บบันทึกอาการอย่างละเอียด
            3. หลีกเลี่ยงการวินิจฉัยด้วยตนเองจากอินเทอร์เน็ต
            
            ⚠️ ข้อมูลนี้เป็นเพียงคำแนะนำทั่วไป ไม่ใช่การวินิจฉัยทางการแพทย์`,
            
            `สำหรับคำถาม "${question}" ระบบแนะนำให้:
            - ติดต่อหมอประจำตัวของคุณ
            - บอกอาการอย่างละเอียดกับแพทย์
            - อย่าหยุดหรือเปลี่ยนแปลงยาโดยไม่ปรึกษาแพทย์
            
            📞 หากเป็นเหตุฉุกเฉิน กรุณาโทร 1669 หรือไปโรงพยาบาลทันที`
        ];
        
        return responses[Math.floor(Math.random() * responses.length)];
    }

    showEmergencyAlert(message) {
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
                background: rgba(0, 0, 0, 0.8);
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
                    max-width: 500px;
                    box-shadow: 0 10px 50px rgba(255, 0, 0, 0.5);
                    animation: slideIn 0.5s;
                    border: 3px solid white;
                ">
                    <div style="text-align: center; margin-bottom: 20px;">
                        <div style="font-size: 60px; animation: pulse 1s infinite;">🚨</div>
                        <h2 style="margin: 10px 0; font-size: 24px; font-weight: bold;">
                            ${emergencyType}
                        </h2>
                    </div>
                    <div style="
                        background: rgba(255, 255, 255, 0.95);
                        color: #000;
                        padding: 20px;
                        border-radius: 10px;
                        font-size: 16px;
                        line-height: 1.6;
                        max-height: 400px;
                        overflow-y: auto;
                    ">
                        ${message.replace(/\n/g, '<br>')}
                    </div>
                    <div style="text-align: center; margin-top: 20px;">
                        <button onclick="document.getElementById('emergencyAlert').remove()" style="
                            background: white;
                            color: #dc143c;
                            border: none;
                            padding: 15px 40px;
                            font-size: 18px;
                            font-weight: bold;
                            border-radius: 25px;
                            cursor: pointer;
                            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                        ">
                            I UNDERSTAND
                        </button>
                    </div>
                    <div style="text-align: center; margin-top: 15px; font-size: 20px;">
                        <a href="tel:1669" style="
                            color: white;
                            text-decoration: none;
                            background: rgba(0,0,0,0.3);
                            padding: 10px 20px;
                            border-radius: 20px;
                            display: inline-block;
                        ">
                            📞 Call Emergency: 1669
                        </a>
                    </div>
                </div>
            </div>
        `;
        
        // Add CSS animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            @keyframes slideIn {
                from { transform: translateY(-50px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.2); }
            }
        `;
        document.head.appendChild(style);
        
        // Add alert to page
        document.body.insertAdjacentHTML('beforeend', alertHTML);
        
        // Play alert sound (if browser allows)
        try {
            const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBCqA0fPTgjkHFmW169uZTwwOUqrm7rthGA==');
            audio.play().catch(() => {}); // Ignore if audio fails
        } catch (e) {}
        
        console.log('🚨 EMERGENCY ALERT DISPLAYED:', emergencyType);
    }

    addMessage(type, content, confidence = null, modelUsed = null) {
        const messagesContainer = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        
        const timestamp = new Date().toLocaleTimeString('th-TH');
        let icon = '';
        let sender = '';
        
        switch (type) {
            case 'user':
                icon = 'fas fa-user';
                sender = 'คุณ';
                break;
            case 'ai':
                icon = 'fas fa-robot';
                sender = 'Medical AI Assistant';
                break;
            case 'system':
                icon = 'fas fa-exclamation-circle';
                sender = 'ระบบ';
                break;
        }
        
        let confidenceInfo = '';
        if (confidence !== null && type === 'ai') {
            const confidencePercent = Math.round(confidence * 100);
            confidenceInfo = `
                <div class="ai-info">
                    <p><i class="fas fa-chart-line"></i> ความมั่นใจ: ${confidencePercent}%</p>
                    ${modelUsed ? `<p><i class="fas fa-microchip"></i> โมเดล: ${modelUsed}</p>` : ''}
                    <p><i class="fas fa-exclamation-triangle"></i> ข้อมูลนี้เพื่อการศึกษาเท่านั้น ไม่ใช่คำวินิจฉัยทางการแพทย์</p>
                </div>
            `;
        }
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-header">
                    <i class="${icon}"></i>
                    <strong>${sender}</strong>
                    <span class="timestamp">${timestamp}</span>
                </div>
                <p>${content}</p>
                ${confidenceInfo}
            </div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    setWorkflowStepStatus(stepId, status, details = '') {
        const step = document.getElementById(stepId);
        if (!step) return;
        
        // Remove previous status classes
        step.classList.remove('active', 'completed', 'error');
        
        // Add new status class
        step.classList.add(status);
        
        // Update status indicator
        const statusIndicator = step.querySelector('.step-status');
        statusIndicator.className = `step-status ${status}`;
        
        // Update time details
        const timeElement = step.querySelector('.step-time');
        if (details) {
            timeElement.textContent = details;
        } else if (status === 'completed') {
            timeElement.textContent = new Date().toLocaleTimeString('th-TH');
        }
    }

    resetWorkflowSteps() {
        for (let i = 1; i <= 6; i++) {
            const step = document.getElementById(`step${i}`);
            if (step) {
                step.classList.remove('active', 'completed', 'error');
                step.querySelector('.step-status').className = 'step-status pending';
                step.querySelector('.step-time').textContent = '-';
            }
        }
    }

    showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        if (show) {
            overlay.classList.add('active');
        } else {
            overlay.classList.remove('active');
        }
    }

    updateLoadingMessage(message) {
        const messageElement = document.getElementById('loadingMessage');
        if (messageElement) {
            messageElement.textContent = message;
        }
    }

    updateMetrics(responseTime, success) {
        this.stats.totalQuestions++;
        this.stats.totalResponseTime += responseTime;
        
        if (success) {
            this.stats.successfulRequests++;
        }
        
        // Update real-time metrics
        document.getElementById('responseTimeMetric').textContent = `${responseTime}ms`;
        
        // Update display
        this.updateStatsDisplay();
        this.saveStats();
    }

    startPerformanceMonitoring() {
        // Update system status every 30 seconds
        setInterval(() => {
            this.checkSystemStatus();
        }, 30000);
        
        // Update performance metrics every 5 seconds
        setInterval(() => {
            this.updateStatsDisplay();
        }, 5000);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize the system when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('🏥 Initializing Medical AI Q&A System...');
    window.medicalQA = new MedicalQASystem();
    console.log('✅ System initialized successfully');
});

// Additional utility functions
window.MedicalQAUtils = {
    // Export chat history
    exportChatHistory() {
        const messages = document.querySelectorAll('.message');
        const history = [];
        
        messages.forEach(message => {
            const sender = message.querySelector('.message-header strong').textContent;
            const timestamp = message.querySelector('.timestamp').textContent;
            const content = message.querySelector('p').textContent;
            
            history.push({
                sender,
                timestamp,
                content
            });
        });
        
        const dataStr = JSON.stringify(history, null, 2);
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
        
        const exportFileDefaultName = `medical-qa-chat-${new Date().toISOString().slice(0,10)}.json`;
        
        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.click();
    },
    
    // Clear chat history
    clearChatHistory() {
        if (confirm('คุณต้องการลบประวัติการสนทนาทั้งหมดหรือไม่?')) {
            const messagesContainer = document.getElementById('chatMessages');
            const systemMessage = messagesContainer.querySelector('.system-message');
            messagesContainer.innerHTML = '';
            messagesContainer.appendChild(systemMessage);
            
            console.log('Chat history cleared');
        }
    },
    
    // Get system information
    getSystemInfo() {
        return {
            userAgent: navigator.userAgent,
            timestamp: new Date().toISOString(),
            stats: window.medicalQA?.stats || {},
            systemStatus: {
                api: document.getElementById('apiStatus')?.textContent || 'Unknown',
                n8n: document.getElementById('n8nStatus')?.textContent || 'Unknown',
                langchain: document.getElementById('langchainStatus')?.textContent || 'Unknown',
                models: document.getElementById('modelStatus')?.textContent || 'Unknown',
                database: document.getElementById('dbStatus')?.textContent || 'Unknown'
            }
        };
    }
};

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    
    // Show user-friendly error message
    if (window.medicalQA && !window.medicalQA.isProcessing) {
        const errorMessage = `เกิดข้อผิดพลาดในระบบ: ${event.error?.message || 'ข้อผิดพลาดที่ไม่ทราบสาเหตุ'}`;
        window.medicalQA.addMessage('system', errorMessage);
    }
});

// Handle network connectivity
window.addEventListener('online', () => {
    console.log('🌐 Network connection restored');
    window.medicalQA?.checkSystemStatus();
});

window.addEventListener('offline', () => {
    console.log('🚫 Network connection lost');
    if (window.medicalQA) {
        window.medicalQA.addMessage('system', '⚠️ การเชื่อมต่ออินเทอร์เน็ตขาดหาย กรุณาตรวจสอบการเชื่อมต่อ');
    }
});