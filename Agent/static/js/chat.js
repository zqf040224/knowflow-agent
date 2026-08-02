        // ==================== 状态管理 ====================
        let currentMode = 'quick';
        let isGenerating = false;
        let currentDocument = '';
        let currentDocumentTemplate = 'auto';
        let currentSpreadsheetTemplate = 'auto';
        let currentEditorSources = [];
        let selectedFile = null;
        let currentUploadMode = 'temp';
        let hasConversationStarted = false;
        let attachedFiles = []; // 已上传的临时文件列表
        let currentSessionId = null; // 当前会话ID
        let sessions = []; // 会话列表
        let isComposing = false; // 输入法状态
        let userScrolledUp = false; // 用户手动上滑后停止自动滚底
        let programmaticScrollUntil = 0;
        let spreadsheetTransformFile = null;

        // ==================== DOM 元素 ====================
        const welcomeState = document.getElementById('welcomeState');
        const chatMessages = document.getElementById('chatMessages');
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        const editorToggle = document.getElementById('editorToggle');
        const editorSection = document.getElementById('editorSection');
        const editorDisplay = document.getElementById('editorDisplay');
        const themeToggle = document.getElementById('themeToggle');

        // ==================== 工具函数 ====================

        function applyTheme(theme) {
            const nextTheme = theme === 'dark' ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', nextTheme);
            localStorage.setItem('theme', nextTheme);
            if (themeToggle) {
                const isDark = nextTheme === 'dark';
                themeToggle.setAttribute('aria-label', isDark ? '切换浅色模式' : '切换深色模式');
                themeToggle.setAttribute('title', isDark ? '切换浅色模式' : '切换深色模式');
                const icon = themeToggle.querySelector('.theme-icon');
                if (icon) icon.textContent = isDark ? '浅色' : '深色';
            }
        }

        function toggleTheme() {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
            applyTheme(currentTheme === 'dark' ? 'light' : 'dark');
        }

        applyTheme(localStorage.getItem('theme') || 'light');
        
        // 防抖函数
        function debounce(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        }

        function getFileExtension(filename) {
            const index = String(filename || '').lastIndexOf('.');
            return index >= 0 ? String(filename).slice(index).toLowerCase() : '';
        }

        function isSpreadsheetFilename(filename) {
            return ['.xlsx', '.xls', '.csv'].includes(getFileExtension(filename));
        }

        function shouldHandleAsSpreadsheetTransform(message, files) {
            if (!files.some(file => file.is_spreadsheet || isSpreadsheetFilename(file.filename))) {
                return false;
            }
            return /(排序|排列|筛选|过滤|导出|降序|升序|从高到低|从低到高|大于|小于|等于|包含|分组|前\s*\d+\s*(条|行|名|个)?)/.test(message);
        }

        function refreshComposerState() {
            if (!sendBtn || !chatInput) return;
            const canSend = Boolean(chatInput.value.trim()) || attachedFiles.length > 0;
            sendBtn.classList.toggle('active', canSend && !isGenerating);
        }

        // 本地存储工具
        const storage = {
            get: (key, defaultValue) => {
                try {
                    const value = localStorage.getItem(key);
                    return value ? JSON.parse(value) : defaultValue;
                } catch (e) {
                    return defaultValue;
                }
            },
            set: (key, value) => {
                try {
                    localStorage.setItem(key, JSON.stringify(value));
                } catch (e) {
                    console.error('Storage error:', e);
                }
            },
            remove: (key) => {
                try {
                    localStorage.removeItem(key);
                } catch (e) {
                    console.error('Storage error:', e);
                }
            }
        };

        // ==================== 输入框自动调整 ====================
        chatInput.addEventListener('input', function() {
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';

            refreshComposerState();
        });

        // 输入法状态监听
        chatInput.addEventListener('compositionstart', function() {
            isComposing = true;
        });

        chatInput.addEventListener('compositionend', function() {
            isComposing = false;
        });

        chatInput.addEventListener('keydown', function(e) {
            // 输入法激活时不发送消息（e.isComposing 是浏览器原生属性，
            // 在 compositionend 之后触发的 keydown 中仍然为 true，避免误发送）
            if (e.isComposing || isComposing) return;

            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if ((this.value.trim() || attachedFiles.length > 0) && !isGenerating) {
                    sendMessage();
                }
            }
        });

        // 模式选择函数
        function selectChatMode(mode) {
            currentMode = mode;
            // 更新按钮状态
            document.querySelectorAll('.mode-selector .mode-btn').forEach(btn => btn.classList.remove('active'));
            const activeBtn = document.querySelector(`.mode-selector .mode-btn[data-mode="${mode}"]`);
            if (activeBtn) {
                activeBtn.classList.add('active');
            }
            // 模式已切换
        }

        // 键盘快捷键
        document.addEventListener('keydown', function(e) {
            // 输入法激活时不发送消息
            if (e.isComposing || isComposing) return;

            // Ctrl/Cmd + Enter 发送消息
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !isGenerating) {
                if (chatInput.value.trim() || attachedFiles.length > 0) {
                    sendMessage();
                }
            }

            // Ctrl/Cmd + N: 新对话
            if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
                e.preventDefault();
                createNewSession();
            }

            // Esc: 关闭编辑器/弹窗
            if (e.key === 'Escape') {
                const feedbackModalEl = document.getElementById('feedbackModal');
                const spreadsheetTransformModalEl = document.getElementById('spreadsheetTransformModal');
                if (feedbackModalEl && feedbackModalEl.classList.contains('show')) {
                    hideFeedbackModal();
                } else if (spreadsheetTransformModalEl && spreadsheetTransformModalEl.classList.contains('show')) {
                    hideSpreadsheetTransformModal();
                } else if (uploadModal && uploadModal.classList.contains('show')) {
                    hideUploadModal();
                } else if (editorSection && editorSection.classList.contains('open')) {
                    toggleEditor();
                }
            }
        });

        // ==================== 消息发送 ====================
        function setInput(text) {
            chatInput.value = text;
            chatInput.focus();
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
            refreshComposerState();
        }

        function showWelcomeState() {
            if (welcomeState) welcomeState.style.display = 'flex';
            if (chatMessages) chatMessages.style.display = 'none';
            hasConversationStarted = false;
            userScrolledUp = false;
        }

        function showChatMessages() {
            if (welcomeState) welcomeState.style.display = 'none';
            if (chatMessages) chatMessages.style.display = 'block';
            hasConversationStarted = true;
        }

        // 保存消息到本地存储
        function saveMessageToStorage(role, content, isDocument = false) {
            if (!currentSessionId) {
                currentSessionId = createSessionId();
            }

            const sessions = storage.get('sessions', []);
            let currentSession = sessions.find(s => s.id === currentSessionId);

            if (!currentSession) {
                currentSession = {
                    id: currentSessionId,
                    title: content.substring(0, 20) + (content.length > 20 ? '...' : ''),
                    messages: [],
                    timestamp: new Date().toISOString()
                };
                sessions.push(currentSession);
            }
            
            // 添加新消息
            currentSession.messages.push({
                role,
                content,
                timestamp: new Date().toISOString(),
                isDocument
            });
            
            // 最多保留三轮对话（每轮包含用户和助手的消息）
            while (currentSession.messages.length > 6) { // 3轮 × 2条消息/轮
                currentSession.messages.shift();
            }
            
            // 更新会话标题和时间戳
            if (role === 'user') {
                currentSession.title = content.substring(0, 20) + (content.length > 20 ? '...' : '');
            }
            currentSession.timestamp = new Date().toISOString();
            
            // 最多保留3个会话
            if (sessions.length > 3) {
                sessions.shift();
            }
            
            // 保存到本地存储
            storage.set('sessions', sessions);
            // 更新会话列表显示
            updateSessionList();
        }

        // 加载历史消息
        async function loadHistoryMessages() {
            await loadSessions();
            if (sessions.length > 0) {
                const latestSession = sessions[sessions.length - 1];
                await loadSession(latestSession.id);
            }
        }

        // 从后端加载会话列表
        async function loadSessions() {
            try {
                const token = localStorage.getItem('token');
                if (!token) return;
                const response = await fetch('/api/sessions', {
                    headers: { 'Authorization': 'Bearer ' + token }
                });
                const data = await response.json();
                if (data.success) {
                    sessions = data.sessions.map(s => ({
                        id: s.session_id,
                        title: s.title || '未命名',
                        messages: [],
                        timestamp: s.updated_at || s.created_at || new Date().toISOString()
                    }));
                }
            } catch (e) {
                // 回退到本地存储
                sessions = storage.get('sessions', []);
            }
            updateSessionList();
        }

        // 更新会话列表显示
        function updateSessionList() {
            const conversationList = document.getElementById('conversationList');
            if (!conversationList) return;
            
            if (sessions.length === 0) {
                conversationList.innerHTML = `
                    <div class="sidebar-empty">
                        <p>暂无对话记录</p>
                        <p>开始一个新任务后，这里会保留最近会话。</p>
                    </div>
                `;
                return;
            }
            
            conversationList.innerHTML = sessions.map(session => {
                const isActive = session.id === currentSessionId;
                const title = escapeHtml(session.title || '未命名对话');
                return `
                    <div class="session-item ${isActive ? 'active' : ''}" onclick="loadSession('${session.id}')">
                        <div class="session-body">
                            <div class="session-topline">
                                <div class="session-time">${formatTime(session.timestamp)}</div>
                            </div>
                            <div class="session-preview">${title}</div>
                            <div class="session-meta">${isActive ? '当前对话' : '点击继续会话'}</div>
                        </div>
                        <button class="session-delete" onclick="event.stopPropagation(); deleteSession('${session.id}')">删除</button>
                    </div>
                `;
            }).join('');
        }

        // 创建会话ID
        function createSessionId() {
            return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        }

        // 加载指定会话（从后端获取消息）
        async function loadSession(sessionId) {
            currentSessionId = sessionId;
            document.body.classList.remove('sidebar-open');

            // 清空当前消息
            if (chatMessages) {
                chatMessages.innerHTML = '';
            }

            try {
                const token = localStorage.getItem('token');
                const response = await fetch(`/api/sessions/${sessionId}`, {
                    headers: { 'Authorization': 'Bearer ' + token }
                });
                const data = await response.json();

                if (data.success && data.messages && data.messages.length > 0) {
                    showChatMessages();
                    editorToggle.classList.remove('hidden');

                    data.messages.forEach(msg => {
                        addMessage(msg.role, msg.content, null, msg.metadata?.attached_files || []);
                    });
                } else {
                    showWelcomeState();
                    editorToggle.classList.remove('hidden');
                }
            } catch (e) {
                // 回退：使用本地存储的消息
                const session = sessions.find(s => s.id === sessionId);
                if (session && session.messages.length > 0) {
                    showChatMessages();
                    editorToggle.classList.remove('hidden');
                    session.messages.forEach(msg => {
                        addMessage(msg.role, msg.content, msg.isDocument ? msg.content : null);
                    });
                } else {
                    showWelcomeState();
                }
            }

            updateSessionList();
        }

        // 删除会话（同步到后端）
        async function deleteSession(sessionId) {
            if (confirm('确定要删除这个对话吗？')) {
                try {
                    const token = localStorage.getItem('token');
                    await fetch(`/api/sessions/${sessionId}`, {
                        method: 'DELETE',
                        headers: { 'Authorization': 'Bearer ' + token }
                    });
                } catch (e) {
                    console.error('删除会话失败:', e);
                }

                sessions = sessions.filter(s => s.id !== sessionId);

                if (currentSessionId === sessionId) {
                    if (sessions.length > 0) {
                        loadSession(sessions[sessions.length - 1].id);
                    } else {
                        currentSessionId = null;
                        if (chatMessages) {
                            chatMessages.innerHTML = '';
                        }
                        showWelcomeState();
                        editorToggle.classList.remove('hidden');
                    }
                }

                updateSessionList();
            }
        }

        // 格式化时间
        function formatTime(timestamp) {
            const date = new Date(timestamp);
            const now = new Date();
            const diff = now - date;
            
            if (diff < 60000) {
                return '刚刚';
            } else if (diff < 3600000) {
                return Math.floor(diff / 60000) + '分钟前';
            } else if (diff < 86400000) {
                return Math.floor(diff / 3600000) + '小时前';
            } else {
                return date.getMonth() + 1 + '-' + date.getDate();
            }
        }

        // 创建新会话（同步到后端）
        async function createNewSession() {
            document.body.classList.remove('sidebar-open');
            // 清空当前消息
            if (chatMessages) {
                chatMessages.innerHTML = '';
            }

            showWelcomeState();
            editorToggle.classList.remove('hidden');

            try {
                const token = localStorage.getItem('token');
                const response = await fetch('/api/sessions', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer ' + token
                    },
                    body: JSON.stringify({ title: '' })
                });
                const data = await response.json();
                if (data.success) {
                    currentSessionId = data.session_id;
                }
            } catch (e) {
                currentSessionId = createSessionId();
            }

            await loadSessions();
        }

        function createMessageAction(label, title, onClick, extraClass = '') {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = `msg-action-btn ${extraClass}`.trim();
            button.textContent = label;
            button.title = title || label;
            button.addEventListener('click', onClick);
            return button;
        }

        function safeCopyText(text, button = null) {
            const done = () => {
                if (button) setTemporaryButtonLabel(button, '已复制', '复制');
            };
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text).then(done).catch(() => fallbackCopyText(text, done));
                return;
            }
            fallbackCopyText(text, done);
        }

        function fallbackCopyText(text, done) {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.setAttribute('readonly', '');
            textarea.style.position = 'fixed';
            textarea.style.left = '-9999px';
            document.body.appendChild(textarea);
            textarea.select();
            try {
                document.execCommand('copy');
                if (done) done();
            } finally {
                textarea.remove();
            }
        }

        function setTemporaryButtonLabel(button, temporaryLabel, originalLabel, duration = 1800) {
            button.textContent = temporaryLabel;
            button.classList.add('copied');
            setTimeout(() => {
                button.textContent = originalLabel;
                button.classList.remove('copied');
            }, duration);
        }

        function revealAnswerArea(answerArea, thinkCard, shouldStream = false) {
            if (!answerArea) return;
            answerArea.hidden = false;
            answerArea.classList.toggle('streaming', Boolean(shouldStream));

            if (thinkCard) {
                thinkCard.classList.add('compact', 'collapsed');
            }
        }

        function draftFollowUp(instruction) {
            if (!chatInput || isGenerating) return;
            chatInput.value = instruction;
            chatInput.focus();
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
            refreshComposerState();
            scrollToBottom(true);
        }

        function retryMessage(message, mode = currentMode) {
            if (!message || isGenerating) return;
            currentMode = mode === 'agent' ? 'agent' : 'quick';
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.mode === currentMode);
            });
            draftFollowUp(message);
            sendMessage();
        }

        function markAnswerImported(answerArea, content) {
            if (!answerArea) return;
            let status = answerArea.querySelector('.answer-import-status');
            const words = (content || '').replace(/\s+/g, '').length;
            if (!status) {
                status = document.createElement('div');
                status.className = 'answer-import-status';
                answerArea.appendChild(status);
            }
            status.innerHTML = `
                <span class="status-dot"></span>
                <span>已导入编辑器 · ${words} 字</span>
            `;
        }

        function isDocumentTaskPrompt(text = '') {
            const compact = String(text || '').replace(/\s+/g, '');
            if (!compact) return false;
            const docTypeHit = /(公文|通知|请示|报告|函|纪要|议案|审议材料|工作方案|工作汇报|对策建议|会议通知|发文|正文|落款)/.test(compact);
            const actionHit = /(写|起草|生成|拟|撰写|整理|改写|转换|排版|导入编辑器)/.test(compact);
            return docTypeHit && actionHit;
        }

        function getAnswerActionPolicy(options = {}) {
            const intent = options.intent || '';
            const hasDocumentContent = Boolean(options.hasDocumentContent || options.document);
            const promptLooksLikeDocument = isDocumentTaskPrompt(options.originalPrompt || options.prompt || '');
            const isDocumentIntent = Boolean(
                options.isDocument ||
                hasDocumentContent ||
                promptLooksLikeDocument ||
                ['doc_drafting', 'doc_formatting'].includes(intent)
            );
            const isKnowledgeIntent = intent === 'knowledge_qa';
            return {
                showEditorActions: isDocumentIntent,
                showEvidenceAction: isDocumentIntent || isKnowledgeIntent
            };
        }

        function parseSseStreamChunk(buffer = '', chunkText = '') {
            const combined = String(buffer || '') + String(chunkText || '');
            const lines = combined.split('\n');
            const nextBuffer = lines.pop();
            const events = [];
            const errors = [];
            lines.forEach(line => {
                if (!line.startsWith('data: ')) return;
                try {
                    events.push(JSON.parse(line.slice(6)));
                } catch (error) {
                    errors.push({ line, message: error.message || String(error) });
                }
            });
            return { events, errors, buffer: nextBuffer };
        }

        function resolveChatDonePayload(data = {}, answerText = '') {
            const resolvedIntent = data.intent || (data.route && data.route.intent) || '';
            const hasDocumentPayload = Boolean(data.document && String(data.document).trim());
            const isDocumentIntent = hasDocumentPayload || ['doc_drafting', 'doc_formatting'].includes(resolvedIntent);
            const finalDoc = isDocumentIntent
                ? (data.document || data.answer || answerText)
                : (data.answer || answerText || data.document || '');
            return {
                resolvedIntent,
                hasDocumentPayload,
                isDocumentIntent,
                finalDoc,
                documentTemplate: isDocumentIntent ? (data.export_template || 'auto') : 'auto',
                spreadsheetTemplate: isDocumentIntent ? (data.export_spreadsheet_template || 'auto') : 'auto'
            };
        }

        if (typeof window !== 'undefined') {
            window.__chatAnswerActionPolicy = getAnswerActionPolicy;
            window.__parseSseStreamChunk = parseSseStreamChunk;
            window.__resolveChatDonePayload = resolveChatDonePayload;
        }

        function appendAnswerActions(answerArea, content, options = {}) {
            if (!answerArea || answerArea.querySelector('.answer-actions')) return;
            const actionsRow = document.createElement('div');
            actionsRow.className = 'message-actions answer-actions';
            const backendActions = Array.isArray(options.actions) ? options.actions : [];
            const policy = getAnswerActionPolicy(options);

            const copyBtn = createMessageAction('复制', '复制内容', function() {
                safeCopyText(content, copyBtn);
            });
            actionsRow.appendChild(copyBtn);

            backendActions.forEach(action => {
                if (!action || !action.type) return;
                const actionBtn = createMessageAction(action.label || '执行操作', action.label || '执行操作', function() {
                    handleBackendAction(action, actionBtn);
                }, 'primary');
                actionsRow.appendChild(actionBtn);
            });

            if (policy.showEditorActions) {
                const importBtn = createMessageAction('导入编辑器', '导入编辑器', function() {
                    insertToEditor(content, {
                        templateType: options.exportTemplate || 'auto',
                        spreadsheetTemplate: options.spreadsheetTemplate || 'auto',
                        sources: Array.isArray(options.sources)
                            ? options.sources
                            : normalizeEditorSources(
                                options.sourceDetails || [],
                                options.sourceFilenames || [],
                                options.sourceFiles || []
                            )
                    });
                    markAnswerImported(answerArea, content);
                    setTemporaryButtonLabel(importBtn, '已导入', '导入编辑器');
                }, backendActions.length ? '' : 'primary');
                actionsRow.appendChild(importBtn);

                const compactBtn = createMessageAction('改短', '基于这条回复生成更精简版本', function() {
                    draftFollowUp('请基于上一条回复改写为更精简的版本，保留关键事实和正式语气。');
                });
                const formalBtn = createMessageAction('更正式', '基于这条回复提升正式程度', function() {
                    draftFollowUp('请基于上一条回复进一步提升正式程度，保持结构不变。');
                });
                actionsRow.appendChild(compactBtn);
                actionsRow.appendChild(formalBtn);
            }

            if (policy.showEvidenceAction) {
                const evidenceBtn = createMessageAction('补充依据', '要求补充知识库依据和来源说明', function() {
                    draftFollowUp('请基于上一条回复补充依据说明，并明确引用知识库来源。');
                });
                actionsRow.appendChild(evidenceBtn);
            }

            if (options.originalPrompt) {
                const regenBtn = createMessageAction('重新生成', '用同一任务重新生成', function() {
                    retryMessage(options.originalPrompt, currentMode);
                });
                actionsRow.appendChild(regenBtn);
            }

            answerArea.appendChild(actionsRow);
        }

        function appendFailureActions(answerArea, message, errorMsg) {
            revealAnswerArea(answerArea, null, false);
            answerArea.classList.add('failure-answer');
            answerArea.innerHTML = `
                <div class="failure-title">请求没有完成</div>
                <div class="failure-copy">${escapeHtml(errorMsg)}</div>
            `;

            const actionsRow = document.createElement('div');
            actionsRow.className = 'message-actions answer-actions';
            actionsRow.appendChild(createMessageAction('重试', '按当前模式重新发送', () => retryMessage(message, currentMode), 'primary'));
            actionsRow.appendChild(createMessageAction('客服重试', '切换为智能客服重新发送', () => retryMessage(message, 'quick')));
            actionsRow.appendChild(createMessageAction('公文协作重试', '切换为公文协作重新发送', () => retryMessage(message, 'agent')));
            const copyBtn = createMessageAction('复制错误', '复制错误信息', () => safeCopyText(errorMsg, copyBtn));
            actionsRow.appendChild(copyBtn);
            answerArea.appendChild(actionsRow);
        }

        function parseTransformSummary(response) {
            const raw = response.headers.get('X-Spreadsheet-Transform-Summary');
            if (!raw) return null;
            try {
                return JSON.parse(decodeURIComponent(raw));
            } catch (error) {
                console.warn('表格处理摘要解析失败:', error);
                return null;
            }
        }

        function renderTransformSummary(summary, filename) {
            const operation = summary?.operation || {};
            const filters = operation.filters || [];
            const sorts = operation.sorts || [];
            const filterText = filters.length
                ? filters.map(item => `${item.column} ${item.operator} ${item.value ?? ''}`).join('；')
                : '无';
            const sortText = sorts.length
                ? sorts.map(item => `${item.column} ${item.direction === 'desc' ? '降序' : '升序'}`).join('；')
                : '无';
            const aiText = operation.used_ai ? 'AI 已解析规则' : '使用规则兜底解析';
            return [
                `已处理表格：${filename}`,
                `原始有效行：${summary?.original_count ?? '-'}，导出行：${summary?.output_count ?? '-'}`,
                `筛选：${filterText}`,
                `排序：${sortText}`,
                aiText
            ].join('\n');
        }

        async function runSpreadsheetTransform(file, instruction, answerArea, thinkCard, thinkSteps) {
            thinkSteps.innerHTML = '';
            _addThinkStep(thinkSteps, '', 'Planner', '正在理解表格处理规则...');
            _addThinkStep(thinkSteps, '', 'Executor', '准备读取表格并执行筛选排序...');

            const response = await fetch('/api/spreadsheets/transform', {
                method: 'POST',
                credentials: 'same-origin',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    file_id: file.file_id,
                    instruction
                }),
                signal: AbortSignal.timeout(180000)
            });

            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                throw new Error(data.message || `表格处理失败（HTTP ${response.status}）`);
            }

            const summary = parseTransformSummary(response);
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const outputName = `${(file.filename || '表格').replace(/[\\/:*?"<>|\r\n\t]+/g, '').slice(0, 40) || '表格'}_处理结果.xlsx`;

            const a = document.createElement('a');
            a.href = url;
            a.download = outputName;
            a.click();

            revealAnswerArea(answerArea, thinkCard, false);
            const summaryText = renderTransformSummary(summary, file.filename || outputName);
            answerArea.innerHTML = `
                <p>${escapeHtml(summaryText).replace(/\n/g, '<br>')}</p>
                <div class="message-actions answer-actions">
                    <a class="msg-action-btn primary" href="${url}" download="${escapeHtml(outputName)}">下载 Excel</a>
                </div>
            `;
            saveMessageToStorage('assistant', summaryText, false);
            return summaryText;
        }

        async function handleBackendAction(action, button) {
            if (!action || !action.type) return;
            const originalLabel = button?.textContent || action.label || '执行操作';
            try {
                if (button) {
                    button.disabled = true;
                    button.textContent = '处理中...';
                }
                if (action.type === 'export_xlsx_template') {
                    await downloadReimbursementTemplate(action);
                    showExportNotification('报销表导出成功');
                } else if (action.type === 'spreadsheet_transform') {
                    await downloadSpreadsheetTransform(action);
                    showExportNotification('表格处理完成');
                } else {
                    throw new Error(`暂不支持的操作：${action.type}`);
                }
                if (button) setTemporaryButtonLabel(button, '已完成', originalLabel);
            } catch (error) {
                console.error('操作失败:', error);
                alert(error.message || '操作失败，请重试');
                if (button) button.textContent = originalLabel;
            } finally {
                if (button) button.disabled = false;
            }
        }

        async function downloadReimbursementTemplate(action) {
            const response = await fetch('/api/export_reimbursement_xlsx', {
                method: 'POST',
                credentials: 'same-origin',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + localStorage.getItem('token')
                },
                body: JSON.stringify({ template_type: action.template_key || action.template || 'auto' })
            });
            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                throw new Error(data.error || `报销表导出失败（HTTP ${response.status}）`);
            }
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filenameFromResponse(response) || '报销表.xlsx';
            a.click();
            URL.revokeObjectURL(url);
        }

        async function downloadSpreadsheetTransform(action) {
            if (!action.file_id) {
                throw new Error('缺少表格附件，请重新上传表格后再试');
            }
            const response = await fetch('/api/spreadsheets/transform', {
                method: 'POST',
                credentials: 'same-origin',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + localStorage.getItem('token')
                },
                body: JSON.stringify({
                    file_id: action.file_id,
                    instruction: action.instruction || ''
                }),
                signal: AbortSignal.timeout(180000)
            });
            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                throw new Error(data.message || `表格处理失败（HTTP ${response.status}）`);
            }
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const outputName = `${(action.filename || '表格').replace(/[\\/:*?"<>|\r\n\t]+/g, '').slice(0, 40) || '表格'}_处理结果.xlsx`;
            const a = document.createElement('a');
            a.href = url;
            a.download = filenameFromResponse(response) || outputName;
            a.click();
            URL.revokeObjectURL(url);
        }

        function getPreviousUserPrompt(messageDiv) {
            let cursor = messageDiv.previousElementSibling;
            while (cursor) {
                if (cursor.classList.contains('user')) {
                    return cursor._messageContent || '';
                }
                cursor = cursor.previousElementSibling;
            }
            return '';
        }

        async function sendMessage() {
            const typedMessage = chatInput.value.trim();
            if ((!typedMessage && attachedFiles.length === 0) || isGenerating) return;
            const message = typedMessage || '请阅读并处理我上传的附件。';

            // 首次发送消息，切换视图
            if (!hasConversationStarted) {
                const skeleton = document.getElementById('skeletonLoader');
                if (skeleton) skeleton.remove();
                showChatMessages();
                editorToggle.classList.remove('hidden');
            }

            // 保存附件列表（发送前）
            const filesToSend = [...attachedFiles];

            // 清空输入框和附件
            chatInput.value = '';
            chatInput.style.height = 'auto';
            clearAttachedFiles(); // 发送后清除附件列表
            refreshComposerState();

            isGenerating = true;
            userScrolledUp = false; // 新消息开始，重置上滑标记

            // 添加用户消息
            addMessage('user', message, null, filesToSend);
            saveMessageToStorage('user', message);

            // Codex 风格：先显示思考过程，完成后折叠，再输出正文。
            const msgId = 'msg-' + Date.now();
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message assistant';
            msgDiv.id = msgId;
            msgDiv.innerHTML = `
                <div class="message-content message-content--stream">
                    <div class="thinking-card" id="think-card-${msgId}">
                        <div class="thinking-card-header" onclick="toggleThinkingCard(this)" title="展开/折叠思考过程">
                            <span class="thinking-title">思考中</span>
                            <span class="thinking-badge"></span>
                        </div>
                        <div class="thinking-card-body">
                            <div class="thinking-summary">正在准备请求...</div>
                            <div class="thinking-steps-inner"></div>
                        </div>
                    </div>
                    <div class="assistant-answer" id="answer-${msgId}" hidden></div>
                </div>
            `;

            chatMessages.appendChild(msgDiv);
            scrollToBottom(true);

            // 引用各个区域
            const thinkCard    = document.getElementById('think-card-' + msgId);
            thinkCard.dataset.startedAt = String(Date.now());
            const thinkHeader  = thinkCard.querySelector('.thinking-card-header');
            const thinkTitle   = thinkHeader.querySelector('.thinking-title');
            const thinkBadge   = thinkHeader.querySelector('.thinking-badge');
            const thinkSteps   = thinkCard.querySelector('.thinking-steps-inner');
            const answerArea   = document.getElementById('answer-' + msgId);

            // 添加首条加载提示
            _addThinkStep(thinkSteps, '', 'System', '正在连接服务器...');

            try {
                // 保存当前会话状态
                storage.set('session_state', {
                    isGenerating: true,
                    currentSessionId: currentSessionId,
                    lastMessage: message
                });

                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer ' + localStorage.getItem('token')
                    },
                    body: JSON.stringify({
                        message,
                        display_message: message,
                        mode: currentMode === 'quick' ? 'quick' : 'agent',
                        session_id: currentSessionId,
                        file_ids: filesToSend.map(f => f.file_id)
                    }),
                    // Agent 完整链路可能包含多轮写作、审核和 R1 反思，不能用 3 分钟硬截断。
                    signal: AbortSignal.timeout(600000)
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                // 清除首条加载提示
                thinkSteps.innerHTML = '';

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let answerText = '';
                let thinkLog = [];
                let contentStarted = false;
                let pendingAnswerDelta = '';
                let renderTimer = null;

                // 定时渲染，避免 rAF 帧率不稳定导致卡顿
                const RENDER_INTERVAL = 24; // ms，让最终正文输出更接近实时流式

                function flushAnswer() {
                    if (!answerText) return;
                    answerArea.innerHTML = formatMessage(answerText);
                    // 智能滚动：只有用户在底部附近才自动滚
                    scrollToBottom(false);
                }

                function markThinkingReadyForAnswer() {
                    if (thinkCard.classList.contains('answer-started')) return;
                    _finishThinking(thinkCard, thinkTitle, thinkBadge, thinkLog.length);
                    thinkCard.classList.add('answer-started');
                }

                async function streamFinalAnswer(finalText) {
                    if (!finalText) return;
                    markThinkingReadyForAnswer();
                    revealAnswerArea(answerArea, null, true);
                    contentStarted = true;
                    const chunkSize = finalText.length > 1800 ? 28 : 14;
                    for (let index = chunkSize; index < finalText.length; index += chunkSize) {
                        answerText = finalText.slice(0, index);
                        flushAnswer();
                        await new Promise(resolve => setTimeout(resolve, 16));
                    }
                    answerText = finalText;
                    flushAnswer();
                }

                function appendAnswerChunk(delta) {
                    if (!delta) return;
                    if (!contentStarted) {
                        contentStarted = true;
                        markThinkingReadyForAnswer();
                        revealAnswerArea(answerArea, null, true);
                    }
                    answerText += delta;
                    if (!renderTimer) {
                        renderTimer = setTimeout(() => {
                            flushAnswer();
                            renderTimer = null;
                        }, RENDER_INTERVAL);
                    }
                }

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const parsedChunk = parseSseStreamChunk(buffer, decoder.decode(value, { stream: true }));
                    buffer = parsedChunk.buffer;
                    parsedChunk.errors.forEach(error => console.error('SSE parse error:', error.message, error.line));

                    for (const data of parsedChunk.events) {
                        try {
                            switch (data.type) {
                                case 'thinking_start':
                                    if (!thinkSteps.querySelector('.thinking-step')) {
                                        _addThinkStep(thinkSteps, '', 'System', data.message || '正在思考');
                                        _updateThinkingBadge(thinkCard, thinkBadge);
                                    }
                                    break;

                                case 'thinking_done':
                                    markThinkingReadyForAnswer();
                                    break;

                                case 'tool_plan': {
                                    const steps = Array.isArray(data.data?.steps) ? data.data.steps : [];
                                    const names = steps.map(step => step.tool).filter(Boolean).join(' -> ');
                                    _addThinkStep(thinkSteps, '', 'TaskPlanner', names ? `工具计划: ${names}` : '已生成工具计划');
                                    _updateThinkingBadge(thinkCard, thinkBadge);
                                    break;
                                }

                                case 'tool_call':
                                    _addThinkStep(thinkSteps, '', 'ToolOrchestrator',
                                        `调用工具: ${data.data?.tool || 'unknown'}${data.data?.requires_confirmation ? ' · 需要确认' : ''}`);
                                    _updateThinkingBadge(thinkCard, thinkBadge);
                                    break;

                                case 'tool_result':
                                    _addThinkStep(thinkSteps, '', 'ToolOrchestrator',
                                        `工具完成: ${data.data?.tool || 'unknown'}`);
                                    _updateThinkingBadge(thinkCard, thinkBadge);
                                    break;

                                case 'tool_confirm_required':
                                    _addThinkStep(thinkSteps, '', 'ToolOrchestrator',
                                        `等待确认: ${data.data?.tool || 'action'}`);
                                    _updateThinkingBadge(thinkCard, thinkBadge);
                                    break;

                                case 'answer_start':
                                    markThinkingReadyForAnswer();
                                    revealAnswerArea(answerArea, null, true);
                                    contentStarted = true;
                                    break;

                                case 'answer_delta':
                                    pendingAnswerDelta = data.data || '';
                                    appendAnswerChunk(pendingAnswerDelta);
                                    break;

                                case 'answer_done':
                                case 'run_done':
                                    break;

                                case 'session':
                                    if (data.session_id) currentSessionId = data.session_id;
                                    break;

                                case 'route':
                                    _addThinkStep(thinkSteps, '', 'IntentRouter',
                                        `意图: ${data.data?.intent || 'knowledge_qa'}${data.data?.reason ? ' · ' + data.data.reason : ''}`);
                                    _updateThinkingBadge(thinkCard, thinkBadge);
                                    break;

                                case 'context_start':
                                    _addThinkStep(thinkSteps, '', 'ContextAgent', data.message || '正在分析上下文');
                                    _updateThinkingBadge(thinkCard, thinkBadge);
                                    break;

                                case 'plan_start':
                                    _addThinkStep(thinkSteps, '', 'Planner', data.message || '正在制定任务计划');
                                    _updateThinkingBadge(thinkCard, thinkBadge);
                                    break;

                                case 'write_start':
                                    _addThinkStep(thinkSteps, '', 'Writer', data.message || '正在生成内容');
                                    _updateThinkingBadge(thinkCard, thinkBadge);
                                    break;

                                case 'content_reset':
                                    // 新一版修订开始，清空之前流式内容，只保留最终版
                                    answerText = '';
                                    contentStarted = true;
                                    markThinkingReadyForAnswer();
                                    revealAnswerArea(answerArea, null, true);
                                    answerArea.innerHTML = `<div class="stream-reset-note">${escapeHtml(data.message || '正在生成修订版本...')}</div>`;
                                    scrollToBottom(false);
                                    break;

                                case 'think':
                                    thinkLog.push(data);
                                    _addThinkStep(thinkSteps, data.emoji, data.agent, data.message);
                                    _updateThinkingBadge(thinkCard, thinkBadge);
                                    break;

                                case 'plan':
                                    _addThinkStep(thinkSteps, '', 'Planner',
                                        '任务类型: ' + (data.data?.task_type || '公文生成') + ' / ' + (data.data?.document_type || '通用公文'));
                                    _updateThinkingBadge(thinkCard, thinkBadge);
                                    break;

                                case 'content':
                                    if (pendingAnswerDelta && pendingAnswerDelta === (data.data || '')) {
                                        pendingAnswerDelta = '';
                                        break;
                                    }
                                    pendingAnswerDelta = '';
                                    // 第一条内容到达 → 思考过程折叠，正文从下方开始流式输出。
                                    appendAnswerChunk(data.data || '');
                                    break;

                                case 'done':
                                    // 确保最后的内容被渲染
                                    if (renderTimer) {
                                        clearTimeout(renderTimer);
                                        renderTimer = null;
                                        flushAnswer();
                                    }

                                    const donePayload = resolveChatDonePayload(data, answerText);
                                    const finalDoc = donePayload.finalDoc;

                                    if (finalDoc) {
                                        if (!contentStarted || !answerText) {
                                            await streamFinalAnswer(finalDoc);
                                        } else {
                                            answerArea.classList.remove('streaming');
                                            if (finalDoc !== answerText) {
                                                // 只有后端提供了修订后的完整稿时才覆盖流式内容。
                                                answerArea.innerHTML = formatMessage(finalDoc);
                                            }
                                            answerText = finalDoc;
                                        }
                                        answerArea.classList.remove('streaming');

                                        const responseDocumentTemplate = donePayload.documentTemplate;
                                        const responseSpreadsheetTemplate = donePayload.spreadsheetTemplate;
                                        const responseEditorSources = donePayload.isDocumentIntent
                                            ? normalizeEditorSources(data.source_details || [], data.source_filenames || [], filesToSend)
                                            : [];
                                        if (donePayload.isDocumentIntent) {
                                            currentDocument = data.document || finalDoc;
                                            currentDocumentTemplate = responseDocumentTemplate;
                                            currentSpreadsheetTemplate = responseSpreadsheetTemplate;
                                            currentEditorSources = responseEditorSources;
                                        } else {
                                            currentDocument = '';
                                            currentDocumentTemplate = 'auto';
                                            currentSpreadsheetTemplate = 'auto';
                                            currentEditorSources = [];
                                        }
                                        syncWordTemplateSelect();
                                        updateSpreadsheetExportButton();
                                        refreshEditorSources();

                                        const sourcePanel = renderSourceDetails(data.source_details || [], data.source_filenames || []);
                                        if (sourcePanel) answerArea.appendChild(sourcePanel);
                                        const auditPanel = renderAuditSummary(data.audit_summary || {});
                                        if (auditPanel) answerArea.appendChild(auditPanel);
                                        appendAnswerActions(answerArea, finalDoc, {
                                            originalPrompt: message,
                                            intent: data.intent || data.route?.intent || '',
                                            actions: data.actions || [],
                                            isDocument: donePayload.isDocumentIntent,
                                            hasDocumentContent: donePayload.hasDocumentPayload,
                                            prompt: message,
                                            exportTemplate: responseDocumentTemplate,
                                            spreadsheetTemplate: responseSpreadsheetTemplate,
                                            sources: responseEditorSources,
                                            sourceDetails: data.source_details || [],
                                            sourceFilenames: data.source_filenames || [],
                                            sourceFiles: filesToSend
                                        });

                                        saveMessageToStorage('assistant', finalDoc, true);
                                    }

                                    _finishThinking(thinkCard, thinkTitle, thinkBadge, thinkLog.length);
                                    answerArea.classList.remove('streaming');
                                    scrollToBottom(true);
                                    break;

                                case 'reasoning_chunk':
                                    // R1 推理链实时流式输出
                                    _appendReasoningChunk(thinkSteps, data.data, msgId);
                                    _updateThinkingBadge(thinkCard, thinkBadge);
                                    break;

                                case 'reflection':
                                    // R1 深度反思完成 — 替换实时推理区为完整反思盒子
                                    _finalizeReflectionBox(thinkSteps, data.data, msgId);
                                    _updateThinkingBadge(thinkCard, thinkBadge);
                                    break;

                                case 'error':
                                    _addThinkStep(thinkSteps, '', 'Error', data.message, true);
                                    thinkTitle.textContent = answerText ? '生成中断' : '生成失败';
                                    thinkBadge.textContent = '';
                                    break;
                            }
                        } catch (e) {
                            console.error('SSE event handling error:', e, data);
                        }
                    }
                }

                // 流结束后兜底：如果没有收到 done 事件但收到了内容，手动完成
                if (renderTimer) {
                    clearTimeout(renderTimer);
                    renderTimer = null;
                }
                if (answerText && contentStarted && !answerArea.querySelector('.answer-actions')) {
                    // done 事件未触发（流异常中断），用已有内容完成渲染
                    answerArea.classList.remove('streaming');
                    answerArea.innerHTML = formatMessage(answerText);
                    _finishThinking(thinkCard, thinkTitle, thinkBadge, thinkLog.length);

                    appendAnswerActions(answerArea, answerText, {
                        originalPrompt: message,
                        prompt: message,
                        isDocument: isDocumentTaskPrompt(message)
                    });

                    currentDocument = '';
                    currentDocumentTemplate = 'auto';
                    currentSpreadsheetTemplate = 'auto';
                    currentEditorSources = [];
                    syncWordTemplateSelect();
                    updateSpreadsheetExportButton();
                    refreshEditorSources();
                    saveMessageToStorage('assistant', answerText, true);
                    scrollToBottom(true);
                }

            } catch (error) {
                console.error('请求失败:', error);
                answerArea.classList.remove('streaming');
                thinkTitle.textContent = '请求失败';

                let errorMsg = '网络错误，请稍后重试';
                if (error.name === 'AbortError') errorMsg = '请求超时，请检查网络连接';
                else if (error.message.includes('401')) {
                    errorMsg = '未授权，请重新登录';
                    setTimeout(() => { localStorage.removeItem('token'); location.href = '/login'; }, 2000);
                } else if (error.message.includes('403')) errorMsg = '权限不足';
                else if (error.message.includes('429')) errorMsg = '请求过于频繁，请稍后再试';

                _addThinkStep(thinkSteps, '', 'Error', errorMsg, true);
                appendFailureActions(answerArea, message, errorMsg);
                saveMessageToStorage('assistant', errorMsg, false);
            } finally {
                isGenerating = false;
                storage.remove('session_state');
                refreshComposerState();
            }
        }

        // 加载知识库分类列表
        async function loadCategories() {
            try {
                const response = await fetch('/api/upload/categories');
                const data = await response.json();
                if (data.success && data.categories) {
                    const select = document.getElementById('categorySelect');
                    select.innerHTML = '<option value="">-- 选择分类 --</option>';
                    data.categories.forEach(cat => {
                        const option = document.createElement('option');
                        option.value = JSON.stringify({category: cat.id, department: cat.department, access_level: cat.access_level});
                        option.textContent = cat.name;
                        select.appendChild(option);
                    });
                }
            } catch (err) {
                console.error('加载分类失败:', err);
            }
        }

        // 初始化：加载会话列表，默认显示欢迎页
        async function initApp() {
            await loadSessions();
            await loadCategories();

            const savedState = storage.get('session_state', null);
            if (savedState && savedState.isGenerating) {
                isGenerating = false;
                storage.remove('session_state');

                const warningDiv = document.createElement('div');
                warningDiv.className = 'message assistant';
                warningDiv.innerHTML = `
                    <div class="message-content">
                        <p>上次会话被中断，可能是因为页面刷新或网络问题。</p>
                        <p>您可以重新发送消息或开始新的对话。</p>
                    </div>
                `;
                if (chatMessages) {
                    showChatMessages();
                    chatMessages.appendChild(warningDiv);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
            }
        }

        // 初始化应用
        initApp();

        // 监听用户手动滚动：上滑时停止自动滚底
        if (chatMessages) {
            chatMessages.addEventListener('scroll', () => {
                if (Date.now() < programmaticScrollUntil) return;
                userScrolledUp = !isChatNearBottom(72);
            }, { passive: true });
        }

        function renderMessageAttachments(files = []) {
            if (!files.length) return '';
            return `
                <div class="message-attachments">
                    ${files.map(file => `
                        <span class="message-attachment-chip">
                            <span class="message-attachment-name">${escapeHtml(file.filename || '附件')}</span>
                            <span class="message-attachment-type">${isSpreadsheetFilename(file.filename) || file.is_spreadsheet ? '表格' : '文档'}</span>
                            ${Number.isFinite(Number(file.char_count)) ? `<span class="message-attachment-meta">${Number(file.char_count)}字</span>` : ''}
                        </span>
                    `).join('')}
                </div>
            `;
        }

        function addMessage(role, content, doc = null, files = []) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv._messageContent = content;

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.innerHTML = `${renderMessageAttachments(files)}${formatMessage(content)}`;

            // 消息操作按钮
            const actions = document.createElement('div');
            actions.className = 'message-actions';

            if (role === 'assistant') {
                const copyBtn = createMessageAction('复制', '复制内容', () => {
                    safeCopyText(content, copyBtn);
                });
                actions.appendChild(copyBtn);

                const regenBtn = createMessageAction('重新生成', '重新生成回复', () => {
                    const previousPrompt = getPreviousUserPrompt(messageDiv);
                    if (!previousPrompt || isGenerating) return;
                    chatInput.value = previousPrompt;
                    chatInput.focus();
                    chatInput.style.height = 'auto';
                    chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
                    refreshComposerState();
                    sendMessage();
                });
                actions.appendChild(regenBtn);
            } else {
                const editBtn = createMessageAction('编辑重发', '重新编辑发送', () => {
                    chatInput.value = content;
                    chatInput.focus();
                    chatInput.style.height = 'auto';
                    chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
                    refreshComposerState();
                });
                actions.appendChild(editBtn);
            }

            if (doc) {
                const importBtn = createMessageAction('导入编辑器', '导入编辑器', () => {
                    insertToEditor(doc);
                    markAnswerImported(contentDiv, doc);
                    setTemporaryButtonLabel(importBtn, '已导入', '导入编辑器');
                }, 'primary');
                actions.appendChild(importBtn);
            }

            contentDiv.appendChild(actions);

            messageDiv.appendChild(contentDiv);

            if (chatMessages) {
                chatMessages.appendChild(messageDiv);
                scrollToBottom(true);
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function normalizeEditorSources(sourceDetails = [], sourceFilenames = [], fallbackFiles = []) {
            const normalized = [];
            const seen = new Set();
            const addSource = (filename, meta = '生成来源') => {
                const name = String(filename || '').trim();
                if (!name || seen.has(name)) return;
                seen.add(name);
                normalized.push({ filename: name, meta });
            };

            sourceDetails.forEach(item => {
                if (!item) return;
                addSource(item.filename || item.name, item.meta || item.source_type || '生成来源');
            });
            sourceFilenames.forEach(name => addSource(name, '生成来源'));
            fallbackFiles.forEach(file => addSource(file.filename || file.name, '本次上传'));

            return normalized;
        }

        function renderSourceDetails(sourceDetails = [], sourceFilenames = []) {
            const normalized = [];
            const seen = new Set();

            sourceDetails.forEach(item => {
                if (!item || !item.filename) return;
                const key = String(item.filename);
                if (seen.has(key)) return;
                seen.add(key);
                normalized.push({ filename: String(item.filename) });
            });

            sourceFilenames.forEach(name => {
                if (!name) return;
                const key = String(name);
                if (seen.has(key)) return;
                seen.add(key);
                normalized.push({ filename: String(name), source_type: 'document' });
            });

            if (!normalized.length) return null;

            const panel = document.createElement('div');
            panel.className = 'source-trace-panel';
            if (normalized.length > 3) {
                panel.classList.add('collapsed');
            }
            const visibleSources = normalized.slice(0, 8);
            const rows = visibleSources.map(item => {
                return `
                    <div class="source-trace-item">
                        <div class="source-trace-main">
                            <span class="source-trace-name">${escapeHtml(item.filename)}</span>
                        </div>
                    </div>
                `;
            }).join('');
            const moreNote = normalized.length > visibleSources.length
                ? `<div class="source-trace-more">仅显示前 ${visibleSources.length} 个，更多来源可在生成记录中查看。</div>`
                : '';

            panel.innerHTML = `
                <button class="source-trace-header" type="button">
                    <span class="source-trace-title">引用来源</span>
                    <span class="source-trace-count">${normalized.length} 个来源</span>
                    <span class="source-trace-toggle">展开</span>
                </button>
                <div class="source-trace-list">${rows}${moreNote}</div>
            `;
            const toggle = panel.querySelector('.source-trace-header');
            const toggleText = panel.querySelector('.source-trace-toggle');
            const syncToggleText = () => {
                if (toggleText) toggleText.textContent = panel.classList.contains('collapsed') ? '展开' : '收起';
            };
            syncToggleText();
            toggle.addEventListener('click', () => {
                panel.classList.toggle('collapsed');
                syncToggleText();
            });
            return panel;
        }

        function renderAuditSummary(audit = {}) {
            if (!audit || !audit.spreadsheet_evidence_count) return null;

            const verified = Array.isArray(audit.verified_claims) ? audit.verified_claims : [];
            const unverified = Array.isArray(audit.unverified_claims) ? audit.unverified_claims : [];
            const issues = Array.isArray(audit.issues) ? audit.issues : [];
            const panel = document.createElement('div');
            panel.className = `audit-summary-panel ${audit.passed ? 'passed' : 'failed'}`;

            const verifiedText = verified.length
                ? verified.slice(0, 8).map(escapeHtml).join('、')
                : '未发现需要校验的数值';
            const unverifiedHtml = unverified.length
                ? `<div class="audit-summary-row"><span>未验证</span><strong>${unverified.slice(0, 8).map(escapeHtml).join('、')}</strong></div>`
                : '';
            const issueHtml = issues.length
                ? `<div class="audit-summary-issue">${issues.slice(0, 2).map(escapeHtml).join('；')}</div>`
                : '';

            panel.innerHTML = `
                <div class="audit-summary-title">${audit.passed ? '报表数值校验通过' : '报表数值校验未通过'}</div>
                <div class="audit-summary-row"><span>已验证</span><strong>${verifiedText}</strong></div>
                ${unverifiedHtml}
                ${issueHtml}
            `;
            return panel;
        }

        // HTML 安全清洗，防止 XSS
        function sanitizeHTML(html) {
            const doc = new DOMParser().parseFromString(html, 'text/html');
            // 移除危险标签
            doc.querySelectorAll('script, iframe, object, embed, link, style').forEach(el => el.remove());
            // 移除所有元素上的事件处理器属性
            doc.querySelectorAll('*').forEach(el => {
                const attrs = [...el.attributes];
                attrs.forEach(attr => {
                    if (attr.name.toLowerCase().startsWith('on')) {
                        el.removeAttribute(attr.name);
                    }
                });
            });
            return doc.body.innerHTML;
        }

        function formatMessage(text) {
            if (typeof marked !== 'undefined') {
                marked.setOptions({ breaks: true, gfm: true });
                const raw = marked.parse(text);
                return sanitizeHTML(raw);
            }
            // 降级：简单正则渲染
            return escapeHtml(text)
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
                .replace(/`([^`]+)`/g, '<code>$1</code>')
                .replace(/\n/g, '<br>');
        }

        // ==================== 编辑器 ====================
        let editorAutoSaveInterval;
        let documentVersions = [];
        let currentVersionIndex = -1;

        function toggleEditor() {
            console.log('toggleEditor called, current state:', editorSection.classList.contains('open'));
            editorSection.classList.toggle('open');
            editorToggle.classList.toggle('hidden');
            document.body.classList.remove('sidebar-open');
            document.body.classList.toggle('editor-open', editorSection.classList.contains('open'));
            
            const mainContainer = document.querySelector('.main-container');
            if (mainContainer) {
                mainContainer.classList.toggle('with-editor', editorSection.classList.contains('open'));
                console.log('Main container class updated:', mainContainer.classList);
            }
            
            if (editorSection.classList.contains('open')) {
                startAutoSave();
                console.log('Editor opened');
            } else {
                stopAutoSave();
                console.log('Editor closed');
            }
        }

        function normalizeDocumentTemplate(templateType) {
            const template = String(templateType || '').trim().toLowerCase();
            return ['auto', 'default', 'review_proposal'].includes(template) ? template : 'auto';
        }

        function detectEditorTemplate(content, requestedTemplate = currentDocumentTemplate) {
            const requested = normalizeDocumentTemplate(requestedTemplate);
            if (requested !== 'auto') return requested;

            const sample = String(content || '').slice(0, 500);
            const reviewMarkers = ['议案', '审议', '办公会', '有关事宜', '此议案需决议', '以上，请审议'];
            return reviewMarkers.some(marker => sample.includes(marker)) ? 'review_proposal' : 'default';
        }

        function applyEditorTemplatePreview(content = '', templateType = currentDocumentTemplate) {
            if (!editorDisplay) return 'default';
            const activeTemplate = detectEditorTemplate(content, templateType);
            const templateClasses = ['template-default', 'template-review-proposal'];
            editorDisplay.classList.remove(...templateClasses);
            editorDisplay.classList.add(`template-${activeTemplate.replace(/_/g, '-')}`);
            editorDisplay.dataset.activeTemplate = activeTemplate;

            const paperPage = editorDisplay.closest('.paper-page');
            if (paperPage) {
                paperPage.classList.remove(...templateClasses);
                paperPage.classList.add(`template-${activeTemplate.replace(/_/g, '-')}`);
                paperPage.dataset.templateLabel = activeTemplate === 'review_proposal' ? '会议审议案' : '普通公文';
            }
            return activeTemplate;
        }

        function renderEditorDocument(content, templateType = currentDocumentTemplate) {
            let formattedContent = content || '';
            const lines = formattedContent.split('\n');
            if (!lines.length) return '';
            const activeTemplate = detectEditorTemplate(formattedContent, templateType);
            const titleClass = activeTemplate === 'review_proposal'
                ? 'document-title document-review-title'
                : 'document-title';

            let processedContent = '';
            let inBody = false;

            for (let i = 0; i < lines.length; i++) {
                const line = lines[i].trim();

                if (i === 0 && line) {
                    processedContent += `<div class="${titleClass}">${escapeHtml(line)}</div>`;
                } else if (line.includes('：') && !inBody) {
                    processedContent += `<div class="document-recipient">${escapeHtml(line)}</div>`;
                    inBody = true;
                } else if (line.match(/^一、|^二、|^三、|^四、|^五、|^六、|^七、|^八、|^九、|^十、/)) {
                    processedContent += `<div class="document-level1">${escapeHtml(line)}</div>`;
                } else if (line.match(/^\（一）|^\（二）|^\（三）|^\（四）|^\（五）/)) {
                    processedContent += `<div class="document-level2">${escapeHtml(line)}</div>`;
                } else if (line.match(/^1\.|^2\.|^3\.|^4\.|^5\./)) {
                    processedContent += `<div class="document-level3">${escapeHtml(line)}</div>`;
                } else if (line.match(/^[\u4e00-\u9fa5]+(局|部|委|办|厅|公司|学院|学校|单位|办公室)$/)) {
                    processedContent += `<div class="document-signature">${escapeHtml(line)}</div>`;
                } else if (line.match(/^\d{4}年\d{1,2}月\d{1,2}日$/)) {
                    processedContent += `<div class="document-date">${escapeHtml(line)}</div>`;
                } else if (line.match(/^附件：/)) {
                    processedContent += `<div class="document-attachment">${escapeHtml(line)}</div>`;
                } else if (line) {
                    processedContent += `<div class="document-body"><p>${escapeHtml(line)}</p></div>`;
                } else {
                    processedContent += '<br>';
                }
            }

            return processedContent;
        }

        function setEditorDisplayContent(content, templateType = currentDocumentTemplate) {
            if (!editorDisplay) return;
            applyEditorTemplatePreview(content, templateType);
            editorDisplay.innerHTML = sanitizeHTML(renderEditorDocument(content, templateType));
        }

        function escapeAttribute(value) {
            return escapeHtml(String(value || '')).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
        }

        function updateAutosaveTime() {
            const autosaveTime = document.getElementById('autosaveTime');
            if (!autosaveTime) return;
            const now = new Date();
            autosaveTime.textContent = now.toLocaleTimeString('zh-CN', { hour12: false });
            updateEditorStatePanel();
        }

        function editorTemplateLabel(template = currentDocumentTemplate, content = editorDisplay?.innerText || currentDocument) {
            const normalized = normalizeDocumentTemplate(template);
            if (normalized === 'default') return '普通公文';
            if (normalized === 'review_proposal') return '会议审议案';
            const detected = detectEditorTemplate(content, normalized);
            return detected === 'review_proposal' ? '自动识别（会议审议案）' : '自动识别（普通公文）';
        }

        function currentEditorSourceCount() {
            return currentEditorSources.length || attachedFiles.length || 0;
        }

        function updateEditorStatePanel() {
            const templateState = document.getElementById('editorTemplateState');
            const contentState = document.getElementById('editorContentState');
            const sourceState = document.getElementById('editorSourceState');
            const savedState = document.getElementById('editorSavedState');
            const content = editorDisplay?.innerText?.trim() || '';
            const autosaveTime = document.getElementById('autosaveTime');

            if (templateState) templateState.textContent = editorTemplateLabel();
            if (contentState) contentState.textContent = content ? '可导出' : '空白';
            if (sourceState) sourceState.textContent = `${currentEditorSourceCount()} 个来源`;
            if (savedState) savedState.textContent = content ? (autosaveTime?.textContent || '已保存') : '尚未保存';
        }

        function setExportWordButtonIdle() {
            const exportBtn = document.getElementById('exportWordBtn');
            if (!exportBtn) return;
            exportBtn.innerHTML = '<span class="word-badge">W</span><span>导出 Word</span><span class="export-caret">⌄</span>';
        }

        function syncWordTemplateSelect() {
            const select = document.getElementById('wordTemplateSelect');
            if (!select) return;
            select.value = normalizeDocumentTemplate(currentDocumentTemplate);
        }

        function setWordExportTemplate(templateType) {
            currentDocumentTemplate = normalizeDocumentTemplate(templateType);
            syncWordTemplateSelect();
            const content = editorDisplay?.innerText || currentDocument || '';
            setEditorDisplayContent(content, currentDocumentTemplate);
            currentDocument = content;
            if (content.trim()) {
                storage.set('editor_content', content);
            }
            updateWordCount();
            updateAutosaveTime();
            updateEditorStatePanel();
        }

        function refreshEditorSources() {
            const sourceList = document.getElementById('editorSourceList');
            if (!sourceList) return;
            const editorSources = currentEditorSources.length
                ? currentEditorSources
                : normalizeEditorSources([], [], attachedFiles);
            if (!editorSources.length) {
                sourceList.innerHTML = '<div class="editor-side-empty">上传附件或生成文档后显示来源</div>';
                updateEditorStatePanel();
                return;
            }
            const items = editorSources.slice(0, 4).map(source => {
                const name = source.filename || '已上传材料';
                const ext = getFileExtension(name);
                const isPdf = ext === '.pdf';
                const isExcel = isSpreadsheetFilename(name);
                const typeLabel = isExcel ? 'Excel' : (isPdf ? 'PDF' : 'Word');
                const iconClass = isExcel ? 'excel' : (isPdf ? 'pdf' : 'word');
                const iconText = isExcel ? 'X' : (isPdf ? 'PDF' : 'W');
                return `
                    <div class="source-item">
                        <span class="source-icon ${iconClass}">${iconText}</span>
                        <span class="source-meta">
                            <strong>${escapeHtml(name)}</strong>
                            <small>${escapeHtml(source.meta || '生成来源')}</small>
                        </span>
                        <span class="source-type ${iconClass}">${typeLabel}</span>
                    </div>
                `;
            }).join('');
            sourceList.innerHTML = sanitizeHTML(items);
            updateEditorStatePanel();
        }

        function insertToEditor(content, options = {}) {
            // 更新编辑器状态
            const editorStatus = document.getElementById('editorStatus');
            if (editorStatus) {
                editorStatus.innerHTML = '<span class="status-icon">↻</span><span class="status-text">生成中...</span>';
                editorStatus.style.background = 'rgba(59, 130, 246, 0.1)';
                editorStatus.style.borderColor = 'rgba(59, 130, 246, 0.3)';
                editorStatus.style.color = '#3b82f6';
            }
            
            currentDocumentTemplate = normalizeDocumentTemplate(options.templateType || 'auto');
            setEditorDisplayContent(content, currentDocumentTemplate);
            if (!editorSection.classList.contains('open')) {
                toggleEditor();
            }
            currentDocument = content;
            currentSpreadsheetTemplate = options.spreadsheetTemplate || 'auto';
            currentEditorSources = Array.isArray(options.sources) ? options.sources : [];
            refreshEditorSources();
            syncWordTemplateSelect();
            updateSpreadsheetExportButton();
            saveDocumentVersion(content);
            updateEditorStatePanel();
            
            // 更新字数统计
            updateWordCount();
            updateAutosaveTime();
            
            // 恢复编辑器状态
            setTimeout(() => {
                if (editorStatus) {
                    editorStatus.innerHTML = '<span class="status-icon">✓</span><span class="status-text">就绪</span>';
                    editorStatus.style.background = 'rgba(16, 185, 129, 0.1)';
                    editorStatus.style.borderColor = 'rgba(16, 185, 129, 0.3)';
                    editorStatus.style.color = '#10b981';
                }
                
                // 显示保存成功提示
                const notification = document.createElement('div');
                notification.className = 'notification';
                notification.innerHTML = '✅ 文档已生成并保存';
                notification.style.cssText = `
                    position: fixed;
                    top: 80px;
                    right: 20px;
                    background: var(--accent-color);
                    color: white;
                    padding: 12px 20px;
                    border-radius: var(--radius-md);
                    box-shadow: var(--shadow-lg);
                    z-index: 1000;
                    animation: slideIn 0.3s ease-out;
                `;
                document.body.appendChild(notification);
                setTimeout(() => {
                    notification.style.animation = 'slideOut 0.3s ease-in forwards';
                    setTimeout(() => notification.remove(), 300);
                }, 2000);
            }, 500);
        }

        // 自动保存
        function startAutoSave() {
            stopAutoSave(); // 清除之前的定时器
            editorAutoSaveInterval = setInterval(() => {
                const content = editorDisplay.innerText;
                if (content.trim() && content !== currentDocument) {
                    currentDocument = content;
                    currentSpreadsheetTemplate = 'auto';
                    applyEditorTemplatePreview(content, currentDocumentTemplate);
                    syncWordTemplateSelect();
                    updateSpreadsheetExportButton();
                    saveDocumentVersion(content);
                    storage.set('editor_content', content);
                    updateAutosaveTime();
                    updateEditorStatePanel();
                }
            }, 30000); // 每30秒自动保存
        }

        function stopAutoSave() {
            if (editorAutoSaveInterval) {
                clearInterval(editorAutoSaveInterval);
                editorAutoSaveInterval = null;
            }
        }

        // 版本控制
        function saveDocumentVersion(content) {
            const normalized = String(content || '').trim();
            if (!normalized) return;
            const lastVersion = documentVersions[documentVersions.length - 1];
            if (lastVersion && lastVersion.content === content) {
                updateVersionControls();
                return;
            }
            // 限制版本数量为10个
            if (documentVersions.length >= 10) {
                documentVersions.shift();
            }
            documentVersions.push({
                content: content,
                timestamp: new Date().toISOString()
            });
            currentVersionIndex = documentVersions.length - 1;
            updateVersionControls();
        }

        function undo() {
            if (currentVersionIndex > 0) {
                currentVersionIndex--;
                const version = documentVersions[currentVersionIndex];
                setEditorDisplayContent(version.content, currentDocumentTemplate);
                currentDocument = version.content;
                syncWordTemplateSelect();
                updateVersionControls();
                updateWordCount();
            }
        }

        function redo() {
            if (currentVersionIndex < documentVersions.length - 1) {
                currentVersionIndex++;
                const version = documentVersions[currentVersionIndex];
                setEditorDisplayContent(version.content, currentDocumentTemplate);
                currentDocument = version.content;
                syncWordTemplateSelect();
                updateVersionControls();
                updateWordCount();
            }
        }

        function updateVersionControls() {
            // 更新版本信息
            const versionInfo = document.getElementById('versionInfo');
            if (versionInfo) {
                versionInfo.textContent = `版本：${currentVersionIndex + 1}/${documentVersions.length}`;
            }
            console.log(`当前版本: ${currentVersionIndex + 1}/${documentVersions.length}`);
        }

        // 字数统计
        function updateWordCount() {
            const content = editorDisplay.innerText;
            const wordCount = content.length;
            const wordCountElement = document.getElementById('wordCount');
            if (wordCountElement) {
                wordCountElement.textContent = `字数：${wordCount}`;
            }
            updateEditorStatePanel();
        }

        if (editorDisplay) {
            editorDisplay.addEventListener('input', debounce(function() {
                const content = editorDisplay.innerText;
                currentDocument = content;
                currentSpreadsheetTemplate = 'auto';
                applyEditorTemplatePreview(content, currentDocumentTemplate);
                syncWordTemplateSelect();
                updateSpreadsheetExportButton();
                if (content.trim()) {
                    storage.set('editor_content', content);
                } else {
                    storage.remove('editor_content');
                }
                updateWordCount();
                updateAutosaveTime();
                updateEditorStatePanel();
            }, 300));
        }

        function editorHasContent() {
            return Boolean(editorDisplay && editorDisplay.innerText.trim());
        }

        function focusEditorForCommand() {
            if (!editorDisplay) return false;
            editorDisplay.focus();
            return true;
        }

        function recordEditorMutation() {
            if (!editorDisplay) return;
            const content = editorDisplay.innerText;
            currentDocument = content;
            currentSpreadsheetTemplate = 'auto';
            applyEditorTemplatePreview(content, currentDocumentTemplate);
            syncWordTemplateSelect();
            updateSpreadsheetExportButton();
            if (content.trim()) {
                saveDocumentVersion(content);
                storage.set('editor_content', content);
            } else {
                storage.remove('editor_content');
            }
            updateWordCount();
            updateAutosaveTime();
            updateEditorStatePanel();
        }

        function formatEditorCommand(command, value = null) {
            if (!focusEditorForCommand()) return;
            document.execCommand(command, false, value);
            recordEditorMutation();
        }

        function changeLineHeight(lineHeight) {
            if (!editorDisplay) return;
            editorDisplay.style.lineHeight = lineHeight;
            localStorage.setItem('editor_line_height', lineHeight);
            updateEditorStatePanel();
        }

        function sourceNameForCitation(fallbackName) {
            if (fallbackName) return fallbackName;
            if (currentEditorSources.length > 0) {
                return currentEditorSources[0].filename || '生成来源';
            }
            const firstSource = document.querySelector('#editorSourceList .source-meta strong');
            if (firstSource?.textContent?.trim()) return firstSource.textContent.trim();
            if (attachedFiles.length > 0) {
                return attachedFiles[0].filename || '上传材料';
            }
            return '';
        }

        function insertSourceCitation(sourceName = '') {
            if (!focusEditorForCommand()) return;
            const name = sourceNameForCitation(sourceName);
            if (!name) {
                alert('暂无可插入的来源。请先上传附件，或使用生成结果中的引用来源。');
                return;
            }
            const citation = `<span class="editor-citation">（来源：${escapeHtml(name)}）</span>`;
            document.execCommand('insertHTML', false, citation);
            recordEditorMutation();
            showExportNotification('已插入来源引用');
        }

        function addEditorComment() {
            if (!focusEditorForCommand()) return;
            const selection = window.getSelection();
            const selectedText = selection ? selection.toString() : '';
            if (selectedText) {
                document.execCommand('insertHTML', false, `<mark class="editor-comment-mark">${escapeHtml(selectedText)}</mark>`);
            } else {
                document.execCommand('insertHTML', false, '<mark class="editor-comment-mark">[批注]</mark>');
            }
            recordEditorMutation();
            showExportNotification('已添加批注标记');
        }

        function toggleReviewDetails(button) {
            const card = button.closest('.review-card');
            if (!card) return;
            let detail = card.querySelector('.review-detail');
            if (!detail) {
                detail = document.createElement('div');
                detail.className = 'review-detail';
                detail.textContent = '该建议可直接应用到当前编辑器内容；最终请以人工复核结果为准。';
                card.appendChild(detail);
            }
            card.classList.toggle('expanded');
            button.textContent = card.classList.contains('expanded') ? '收起详情 ›' : '查看详情 ›';
        }

        function applyReviewSuggestion() {
            if (!focusEditorForCommand()) return;
            const title = editorDisplay.querySelector('.document-title');
            if (title) {
                title.style.textAlign = 'center';
            }
            const firstCard = document.querySelector('.review-card');
            if (firstCard) {
                firstCard.classList.add('is-applied');
                const badge = firstCard.querySelector('em');
                if (badge) {
                    badge.textContent = '已应用';
                    badge.className = 'good';
                }
            }
            recordEditorMutation();
            showExportNotification('已应用格式建议');
        }

        function refreshReviewPanel() {
            updateWordCount();
            showExportNotification('审核面板已刷新');
        }

        function switchEditorSideTab(tabName) {
            document.querySelectorAll('[data-review-tab]').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.reviewTab === tabName);
            });
            const header = document.getElementById('reviewHeader');
            const list = document.getElementById('reviewList');
            const actions = document.querySelector('.review-actions');
            const note = document.querySelector('.review-note');
            if (!header || !list) return;

            if (tabName === 'sources') {
                header.innerHTML = '<strong>引用来源 <span>可插入</span></strong><button type="button" onclick="refreshReviewPanel()">↻ 刷新</button>';
                const sources = currentEditorSources.length
                    ? currentEditorSources
                    : normalizeEditorSources([], [], attachedFiles);
                list.innerHTML = sources.length
                    ? sources.map(source => `
                        <div class="review-source-row">
                            <span><strong>${escapeHtml(source.filename)}</strong><small>${escapeHtml(source.meta || '生成来源')}</small></span>
                            <button type="button" data-source="${escapeAttribute(source.filename)}" onclick="insertSourceCitation(this.dataset.source)">插入</button>
                        </div>
                    `).join('')
                    : '<div class="review-detail" style="display:block;">暂无来源。上传附件或导入生成文档后会显示真实来源。</div>';
                if (actions) actions.style.display = 'none';
                if (note) note.textContent = '引用会插入到当前光标位置。';
                return;
            }

            if (tabName === 'versions') {
                header.innerHTML = '<strong>版本记录 <span>' + documentVersions.length + ' 个</span></strong><button type="button" onclick="refreshReviewPanel()">↻ 刷新</button>';
                if (!documentVersions.length) {
                    list.innerHTML = '<div class="review-detail" style="display:block;">暂无版本记录，生成或编辑文档后会自动保存。</div>';
                } else {
                    list.innerHTML = documentVersions.map((version, index) => {
                        const time = new Date(version.timestamp).toLocaleTimeString('zh-CN', { hour12: false });
                        return `
                            <div class="review-version-row">
                                <span><strong>版本 ${index + 1}</strong><small>${time} · ${version.content.length} 字</small></span>
                                <button type="button" onclick="restoreEditorVersion(${index})">恢复</button>
                            </div>
                        `;
                    }).join('');
                }
                if (actions) actions.style.display = 'none';
                if (note) note.textContent = '版本恢复后仍可继续撤销或重新导出。';
                return;
            }

            header.innerHTML = '<strong>审核建议 <span>0 项</span></strong><button type="button" onclick="refreshReviewPanel()">↻ 刷新</button>';
            list.innerHTML = '<div class="review-detail" style="display:block;">简化版编辑器暂不展示自动审核建议，请以导出的 Word/Excel 文件人工复核为准。</div>';
            if (actions) actions.style.display = 'none';
            if (note) note.textContent = '当前为内测简化版';
        }

        function restoreEditorVersion(index) {
            const version = documentVersions[index];
            if (!version) return;
            currentVersionIndex = index;
            setEditorDisplayContent(version.content, currentDocumentTemplate);
            currentDocument = version.content;
            syncWordTemplateSelect();
            updateVersionControls();
            updateWordCount();
            showExportNotification(`已恢复版本 ${index + 1}`);
        }

        // 思考卡片折叠/展开
        function toggleThinkingCard(headerEl) {
            const card = headerEl.closest('.thinking-card');
            if (!card) return;
            card.classList.toggle('collapsed');
        }

        function _scrollThinkingIntoView(container, force) {
            const body = container?.closest('.thinking-card-body');
            if (body) {
                body.scrollTop = body.scrollHeight;
            }
            scrollToBottom(Boolean(force));
        }

        // R1 推理链实时流式追加
        function _appendReasoningChunk(container, chunkText, msgId) {
            // 查找或创建实时推理容器（不重复创建 think step，由 orchestrator 的 think 事件负责）
            let liveBox = document.getElementById('r1-live-' + msgId);
            if (!liveBox) {
                const wrapper = document.createElement('div');
                wrapper.innerHTML = `
                    <div class="reflection-box" id="r1-live-${msgId}">
                        <div class="reflection-box-header reflection-live-header" style="cursor:default;">
                            <span>深度校验</span>
                            <span class="reflection-toggle-icon">分析中...</span>
                        </div>
                        <div class="reflection-box-body">
                            <div class="reflection-reasoning is-live" id="r1-reasoning-${msgId}"></div>
                        </div>
                    </div>
                `;
                container.appendChild(wrapper.firstElementChild);
                liveBox = document.getElementById('r1-live-' + msgId);
            }

            const reasoningDiv = document.getElementById('r1-reasoning-' + msgId);
            if (reasoningDiv) {
                reasoningDiv.textContent += chunkText;
                reasoningDiv.scrollTop = reasoningDiv.scrollHeight;
            }
            _scrollThinkingIntoView(container, false);
        }

        // R1 反思完成 — 替换实时容器为完整的反思盒子
        function _finalizeReflectionBox(container, meta, msgId) {
            // 移除实时推理容器
            const liveBox = document.getElementById('r1-live-' + msgId);
            if (liveBox) {
                liveBox.remove();
            }

            // 更新最后一个 Reflection 思考步骤的状态
            const steps = container.querySelectorAll('.thinking-step');
            if (steps.length > 0) {
                const lastStep = steps[steps.length - 1];
                const agentEl = lastStep.querySelector('.step-agent');
                const textEl = lastStep.querySelector('.step-text');
                if (agentEl && textEl && lastStep.dataset.agent === 'Reflection') {
                    textEl.textContent = '深度校验完成，详情见下方';
                }
            }

            // 渲染完整反思盒子
            _renderReflectionBox(container, meta);
            _scrollThinkingIntoView(container, false);
        }

        // 渲染 R1 深度反思详情盒子
        function _renderReflectionBox(container, meta) {
            const weaknesses = meta.weaknesses || [];
            const counterArgs = meta.counter_arguments || [];
            const missingEvidence = meta.missing_evidence || [];
            const betterAngle = meta.better_angle || '';
            const logicScore = meta.logic_score || 0.7;
            const reasoning = meta.reasoning_content || '';
            const scorePercent = Math.round(logicScore * 100);

            // 根据分数选颜色
            let scoreColor = '#f59e0b';
            if (scorePercent >= 85) scoreColor = '#22c55e';
            else if (scorePercent < 70) scoreColor = '#ef4444';

            let html = '<div class="reflection-box" id="reflection-box-' + container.closest('.thinking-card')?.id?.replace('think-card-', '') + '">';
            html += '<div class="reflection-box-header" onclick="this.parentElement.classList.toggle(\'collapsed\')">';
            html += '<span>深度校验结果</span>';
            html += '<span class="reflection-toggle-icon">收起</span>';
            html += '</div>';
            html += '<div class="reflection-box-body">';

            // 逻辑评分条
            html += '<div class="reflection-section">';
            html += '<div class="reflection-label">逻辑严谨度</div>';
            html += '<div class="reflection-score-wrap">';
            html += '<div class="reflection-score-bar"><div class="reflection-score-fill" style="width:' + scorePercent + '%;background:' + scoreColor + ';"></div></div>';
            html += '<span class="reflection-score-num" style="color:' + scoreColor + '">' + scorePercent + '</span>';
            html += '</div></div>';

            if (weaknesses.length) {
                html += '<div class="reflection-section">';
                html += '<div class="reflection-label">需要关注</div><ul>';
                weaknesses.forEach(w => { html += '<li>' + escapeHtml(w) + '</li>'; });
                html += '</ul></div>';
            }

            if (counterArgs.length) {
                html += '<div class="reflection-section">';
                html += '<div class="reflection-label">反向校验</div><ul>';
                counterArgs.forEach(c => { html += '<li>' + escapeHtml(c) + '</li>'; });
                html += '</ul></div>';
            }

            if (missingEvidence.length) {
                html += '<div class="reflection-section">';
                html += '<div class="reflection-label">论据补充</div><ul>';
                missingEvidence.forEach(e => { html += '<li>' + escapeHtml(e) + '</li>'; });
                html += '</ul></div>';
            }

            if (betterAngle) {
                html += '<div class="reflection-section">';
                html += '<div class="reflection-label">优化角度</div>';
                html += '<div class="reflection-note">' + escapeHtml(betterAngle) + '</div>';
                html += '</div>';
            }

            // R1 原始推理链
            if (reasoning) {
                html += '<details class="reflection-section reflection-raw" open>';
                html += '<summary class="reflection-label">查看详细推理</summary>';
                html += '<div class="reflection-reasoning">' + escapeHtml(reasoning) + '</div>';
                html += '</details>';
            }

            html += '</div></div>';

            const wrapper = document.createElement('div');
            wrapper.innerHTML = html;
            container.appendChild(wrapper.firstElementChild);
            _scrollThinkingIntoView(container, false);
        }

        // 添加一条思考步骤
        function _addThinkStep(container, emoji, agent, message, isError) {
            const lastStep = container.lastElementChild;
            if (lastStep && lastStep.dataset.agent === agent && lastStep.dataset.message === message) {
                return;
            }

            container.querySelectorAll('.thinking-step.current').forEach(step => {
                step.classList.remove('current');
                step.classList.add('done');
            });

            const div = document.createElement('div');
            const agentInfo = _normalizeAgentInfo(agent);
            div.className = 'thinking-step current ' + agentInfo.className + (isError ? ' error-step' : '');
            div.dataset.agent = agent;
            div.dataset.message = message;
            div.innerHTML = `
                <span class="step-agent">${escapeHtml(agentInfo.label)}</span>
                <span class="step-text">${escapeHtml(message)}</span>
            `;
            container.appendChild(div);

            const summary = container.closest('.thinking-card-body')?.querySelector('.thinking-summary');
            if (summary) {
                summary.textContent = isError ? '执行遇到问题' : message;
            }

            _scrollThinkingIntoView(container, false);
        }

        // 思考完成，折叠卡片
        function _finishThinking(card, titleEl, badgeEl, stepCount) {
            const visibleStepCount = card.querySelectorAll('.thinking-step').length || stepCount || 0;
            titleEl.textContent = '思考过程已折叠';
            const elapsed = _formatThinkingElapsed(card);
            badgeEl.textContent = [
                visibleStepCount > 0 ? visibleStepCount + ' 步' : '',
                elapsed
            ].filter(Boolean).join(' · ');
            card.classList.add('complete');
            card.querySelectorAll('.thinking-step.current').forEach(step => {
                step.classList.remove('current');
                step.classList.add('done');
            });
            const summary = card.querySelector('.thinking-summary');
            if (summary) {
                summary.textContent = '思考完成，正文开始输出';
            }
            card.classList.add('compact', 'collapsed');
            scrollToBottom(false);
        }

        function _formatThinkingBadge(stepCount) {
            return stepCount > 0 ? '已处理 ' + stepCount + ' 步' : '';
        }

        function _formatThinkingElapsed(card) {
            const startedAt = Number(card?.dataset?.startedAt || 0);
            if (!startedAt) return '';
            const seconds = Math.max(1, Math.round((Date.now() - startedAt) / 1000));
            const minutes = Math.floor(seconds / 60);
            const rest = seconds % 60;
            return minutes > 0 ? `${minutes}m ${rest}s` : `${seconds}s`;
        }

        function _updateThinkingBadge(card, badgeEl) {
            const stepCount = card.querySelectorAll('.thinking-step').length;
            badgeEl.textContent = _formatThinkingBadge(stepCount);
        }

        function _normalizeAgentInfo(agent) {
            const map = {
                System: ['SYS', '连接服务', 'agent-system'],
                Orchestrator: ['ORC', '流程编排', 'agent-orchestrator'],
                ContextAgent: ['CTX', '上下文', 'agent-context'],
                Planner: ['PLN', '任务规划', 'agent-planner'],
                SearchAgent: ['WEB', '联网搜索', 'agent-search'],
                KnowledgeAgent: ['KB', '知识库', 'agent-knowledge'],
                Knowledge: ['KB', '知识库', 'agent-knowledge'],
                Writer: ['WRT', '起草生成', 'agent-writer'],
                Reviewer: ['CHK', '质量检查', 'agent-reviewer'],
                Reflection: ['R1', '深度校验', 'agent-reflection'],
                Error: ['ERR', '错误', 'agent-error']
            };
            const item = map[agent] || [String(agent || 'AGT').slice(0, 3).toUpperCase(), agent || 'Agent', 'agent-generic'];
            return {
                shortName: item[0],
                label: item[1],
                className: item[2]
            };
        }

        function isChatNearBottom(threshold = 48) {
            if (!chatMessages) return true;
            return chatMessages.scrollHeight - chatMessages.scrollTop - chatMessages.clientHeight <= threshold;
        }

        function setChatScrollToBottom() {
            if (!chatMessages) return;
            programmaticScrollUntil = Date.now() + 180;
            chatMessages.scrollTop = chatMessages.scrollHeight;
            requestAnimationFrame(() => {
                if (!chatMessages || userScrolledUp) return;
                programmaticScrollUntil = Date.now() + 180;
                chatMessages.scrollTop = chatMessages.scrollHeight;
            });
        }

        // 智能滚动：force=true 强制滚底（重置用户上滑标记），否则仅当用户未手动上滑才滚
        function scrollToBottom(force) {
            if (!chatMessages) return;
            if (force) {
                userScrolledUp = false;
                setChatScrollToBottom();
                return;
            }
            if (userScrolledUp) return;
            setChatScrollToBottom();
        }

        function copyDocument() {
            const content = editorDisplay.innerText;
            navigator.clipboard.writeText(content).then(() => {
                // 显示更友好的提示
                const notification = document.createElement('div');
                notification.className = 'notification';
                notification.innerHTML = '✅ 已复制到剪贴板';
                notification.style.cssText = `
                    position: fixed;
                    top: 80px;
                    right: 20px;
                    background: var(--accent-color);
                    color: white;
                    padding: 12px 20px;
                    border-radius: var(--radius-md);
                    box-shadow: var(--shadow-lg);
                    z-index: 1000;
                    animation: slideIn 0.3s ease-out;
                `;
                document.body.appendChild(notification);
                setTimeout(() => {
                    notification.style.animation = 'slideOut 0.3s ease-in forwards';
                    setTimeout(() => notification.remove(), 300);
                }, 2000);
            }).catch(err => {
                console.error('复制失败:', err);
                alert('复制失败，请手动复制');
            });
        }

        function clearEditor() {
            editorDisplay.innerHTML = '';
            currentDocument = '';
            currentDocumentTemplate = 'auto';
            currentSpreadsheetTemplate = 'auto';
            currentEditorSources = [];
            applyEditorTemplatePreview('', currentDocumentTemplate);
            syncWordTemplateSelect();
            updateSpreadsheetExportButton();
            refreshEditorSources();
            documentVersions = [];
            currentVersionIndex = -1;
            storage.remove('editor_content');
            updateVersionControls();
            updateWordCount();
            updateEditorStatePanel();
        }

        async function exportDocument() {
            const content = editorDisplay.innerText;
            if (!content.trim()) {
                alert('编辑器为空');
                return;
            }

            try {
                // 显示导出中状态
                const exportBtn = document.getElementById('exportWordBtn') || document.querySelector('.editor-btn.primary');
                exportBtn.innerHTML = '<div class="loading-spinner"></div><span>导出中...</span><span class="export-caret">⌄</span>';
                exportBtn.disabled = true;

                const response = await fetch('/api/export_docx', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content, template_type: currentDocumentTemplate || 'auto' })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `公文_${new Date().toISOString().slice(0,10)}.docx`;
                a.click();
                URL.revokeObjectURL(url);

                showExportNotification('导出成功');

            } catch (error) {
                console.error('导出失败:', error);
                alert('导出失败，请重试');
            } finally {
                const exportBtn = document.getElementById('exportWordBtn') || document.querySelector('.editor-btn.primary');
                setExportWordButtonIdle();
                exportBtn.disabled = false;
            }
        }

        async function exportSpreadsheetDocument() {
            const content = editorDisplay.innerText;
            const hasReimbursementTemplate = currentSpreadsheetTemplate && currentSpreadsheetTemplate !== 'auto';
            if (!content.trim() && !hasReimbursementTemplate) {
                alert('编辑器为空');
                return;
            }

            const exportBtn = document.getElementById('exportExcelBtn');
            try {
                if (exportBtn) {
                    exportBtn.innerHTML = '<div class="loading-spinner"></div> 导出中...';
                    exportBtn.disabled = true;
                }

                const endpoint = hasReimbursementTemplate ? '/api/export_reimbursement_xlsx' : '/api/export_xlsx';
                const payload = hasReimbursementTemplate
                    ? { content, template_type: currentSpreadsheetTemplate }
                    : { content };
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filenameFromResponse(response) || `${hasReimbursementTemplate ? '报销表' : '表格'}_${new Date().toISOString().slice(0,10)}.xlsx`;
                a.click();
                URL.revokeObjectURL(url);
                showExportNotification(hasReimbursementTemplate ? '报销表导出成功' : 'Excel 导出成功');
            } catch (error) {
                console.error('Excel 导出失败:', error);
                alert('Excel 导出失败，请重试');
            } finally {
                if (exportBtn) {
                    setSpreadsheetExportButtonIdle();
                    exportBtn.disabled = false;
                }
            }
        }

        function spreadsheetExportButtonText() {
            return currentSpreadsheetTemplate && currentSpreadsheetTemplate !== 'auto' ? '导出报销表' : '导出 Excel';
        }

        function updateSpreadsheetExportButton() {
            const exportBtn = document.getElementById('exportExcelBtn');
            if (exportBtn && !exportBtn.disabled) {
                setSpreadsheetExportButtonIdle();
                exportBtn.title = currentSpreadsheetTemplate && currentSpreadsheetTemplate !== 'auto'
                    ? '按公共资料模板导出报销表'
                    : '导出 Excel 表格';
            }
        }

        function setSpreadsheetExportButtonIdle() {
            const exportBtn = document.getElementById('exportExcelBtn');
            if (!exportBtn) return;
            exportBtn.innerHTML = `<span class="excel-badge">X</span><span>${spreadsheetExportButtonText()}</span>`;
        }

        function filenameFromResponse(response) {
            const disposition = response.headers.get('Content-Disposition') || response.headers.get('content-disposition') || '';
            const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
            if (utf8Match) {
                try {
                    return decodeURIComponent(utf8Match[1]);
                } catch (e) {
                    return utf8Match[1];
                }
            }
            const plainMatch = disposition.match(/filename="?([^";]+)"?/i);
            return plainMatch ? plainMatch[1] : '';
        }

        function showExportNotification(message) {
            const notification = document.createElement('div');
            notification.className = 'notification';
            notification.innerHTML = `✅ ${escapeHtml(message)}`;
            notification.style.cssText = `
                position: fixed;
                top: 80px;
                right: 20px;
                background: var(--accent-color);
                color: white;
                padding: 12px 20px;
                border-radius: var(--radius-md);
                box-shadow: var(--shadow-lg);
                z-index: 1000;
                animation: slideIn 0.3s ease-out;
            `;
            document.body.appendChild(notification);
            setTimeout(() => {
                notification.style.animation = 'slideOut 0.3s ease-in forwards';
                setTimeout(() => notification.remove(), 300);
            }, 2000);
        }

        // 初始化编辑器内容
        function initEditor() {
            editorDisplay.innerHTML = '';
            currentDocument = '';
            currentDocumentTemplate = 'auto';
            currentSpreadsheetTemplate = 'auto';
            currentEditorSources = [];
            documentVersions = [];
            currentVersionIndex = -1;
            storage.remove('editor_content');
            applyEditorTemplatePreview('', currentDocumentTemplate);
            updateWordCount();

            // 加载保存的字体设置
            const savedFontFamily = localStorage.getItem('editor_font_family') || 'SimSun';
            const savedFontSize = localStorage.getItem('editor_font_size') || '16px';
            const savedLineHeight = localStorage.getItem('editor_line_height') || '2.05';

            editorDisplay.style.fontFamily = savedFontFamily;
            editorDisplay.style.fontSize = savedFontSize;
            editorDisplay.style.lineHeight = savedLineHeight;

            // 设置下拉菜单的值
            const fontFamilySelect = document.getElementById('fontFamilySelect');
            const fontSizeSelect = document.getElementById('fontSizeSelect');
            const lineHeightSelect = document.querySelector('.line-select');
            if (fontFamilySelect) fontFamilySelect.value = savedFontFamily;
            if (fontSizeSelect) fontSizeSelect.value = savedFontSize;
            if (lineHeightSelect) lineHeightSelect.value = savedLineHeight;
            syncWordTemplateSelect();
            updateSpreadsheetExportButton();
            refreshEditorSources();
            updateVersionControls();
            updateEditorStatePanel();
        }

        // 字体和大小调整函数
        function changeFontFamily(fontFamily) {
            editorDisplay.style.fontFamily = fontFamily;
            // 保存到本地存储
            localStorage.setItem('editor_font_family', fontFamily);
            updateEditorStatePanel();
        }

        function changeFontSize(fontSize) {
            editorDisplay.style.fontSize = fontSize;
            // 保存到本地存储
            localStorage.setItem('editor_font_size', fontSize);
        }

        // 初始化编辑器
        initEditor();

        // ==================== 上传功能 ====================
        const feedbackModal = document.getElementById('feedbackModal');
        const feedbackSubmitBtn = document.getElementById('feedbackSubmitBtn');
        const uploadModal = document.getElementById('uploadModal');
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const selectedFileDiv = document.getElementById('selectedFile');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const uploadSubmitBtn = document.getElementById('uploadSubmitBtn');
        const spreadsheetTransformModal = document.getElementById('spreadsheetTransformModal');
        const spreadsheetTransformInstruction = document.getElementById('spreadsheetTransformInstruction');

        // 支持多文件上传
        let selectedFiles = [];

        function showInlineNotice(message) {
            const notification = document.createElement('div');
            notification.className = 'notification';
            notification.textContent = message;
            notification.style.cssText = `
                position: fixed;
                top: 80px;
                right: 20px;
                background: var(--accent-color);
                color: white;
                padding: 12px 20px;
                border-radius: var(--radius-md);
                box-shadow: var(--shadow-lg);
                z-index: 1000;
                animation: slideIn 0.3s ease-out;
            `;
            document.body.appendChild(notification);
            setTimeout(() => {
                notification.style.animation = 'slideOut 0.3s ease-in forwards';
                setTimeout(() => notification.remove(), 300);
            }, 2200);
        }

        function sleep(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }

        async function waitForUploadJob(jobId) {
            for (let attempt = 0; attempt < 300; attempt += 1) {
                const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
                    credentials: 'same-origin'
                });
                if (response.status === 401) {
                    setTimeout(() => { localStorage.removeItem('token'); location.href = '/login'; }, 1200);
                    throw new Error('登录已过期，请重新登录');
                }
                const data = await response.json().catch(() => ({}));
                if (!response.ok || !data.success || !data.job) {
                    throw new Error(data.message || `任务查询失败（HTTP ${response.status}）`);
                }
                if (data.job.status === 'succeeded' || data.job.status === 'failed') {
                    return data.job;
                }
                await sleep(2000);
            }
            throw new Error('任务仍在运行，请稍后刷新查看结果');
        }

        function showFeedbackModal() {
            if (!feedbackModal) return;
            document.getElementById('feedbackContent').value = '';
            feedbackModal.classList.add('show');
            setTimeout(() => document.getElementById('feedbackContent')?.focus(), 40);
        }

        function hideFeedbackModal() {
            feedbackModal?.classList.remove('show');
        }

        async function submitFeedback() {
            const contentEl = document.getElementById('feedbackContent');
            const content = contentEl.value.trim();
            if (!content) {
                showInlineNotice('请先填写反馈内容');
                contentEl.focus();
                return;
            }

            feedbackSubmitBtn.disabled = true;
            feedbackSubmitBtn.textContent = '提交中...';

            try {
                const response = await fetch('/api/feedback', {
                    method: 'POST',
                    credentials: 'same-origin',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        category: document.getElementById('feedbackCategory').value,
                        rating: document.getElementById('feedbackRating').value,
                        content,
                        session_id: currentSessionId || '',
                        page_url: window.location.href,
                        context: {
                            mode: currentMode,
                            has_document: Boolean(currentDocument && currentDocument.trim()),
                            attached_file_count: attachedFiles.length,
                            has_conversation_started: hasConversationStarted,
                        },
                    }),
                });
                const data = await response.json().catch(() => ({}));
                if (!response.ok || !data.success) {
                    throw new Error(data.message || '反馈提交失败');
                }
                hideFeedbackModal();
                showInlineNotice('反馈已提交，感谢参与内测');
            } catch (error) {
                console.error('反馈提交失败:', error);
                showInlineNotice(error.message || '反馈提交失败');
            } finally {
                feedbackSubmitBtn.disabled = false;
                feedbackSubmitBtn.textContent = '提交反馈';
            }
        }

        function showUploadModal(mode = 'knowledge') {
            uploadModal.classList.add('show');
            clearFiles();
            selectUploadMode(mode === 'temp' ? 'temp' : 'knowledge');
        }

        function hideUploadModal() {
            uploadModal.classList.remove('show');
        }

        function selectUploadMode(mode) {
            currentUploadMode = mode;
            document.querySelectorAll('.mode-option').forEach(el => el.classList.remove('selected'));
            document.getElementById(`mode${mode.charAt(0).toUpperCase() + mode.slice(1)}`).classList.add('selected');

            // 自动选中 radio
            document.querySelectorAll('input[name="uploadMode"]').forEach(radio => {
                radio.checked = radio.value === mode;
            });
        }

        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = Array.from(e.dataTransfer.files);
            if (files.length > 0) {
                files.forEach(file => handleFile(file));
            }
        });

        fileInput.addEventListener('change', () => {
            const files = Array.from(fileInput.files);
            if (files.length > 0) {
                files.forEach(file => handleFile(file));
            }
        });

        // 生成文件图标
        function getFileIcon(ext) {
            const icons = {
                '.doc': '文档',
                '.docx': '文档',
                '.pdf': 'PDF',
                '.txt': '文本',
                '.md': '文本',
                '.xlsx': '表格',
                '.xls': '表格',
                '.csv': '表格'
            };
            return icons[ext] || '文件';
        }

        function handleFile(file) {
            const validTypes = ['.doc', '.docx', '.pdf', '.txt', '.md', '.xlsx', '.xls', '.csv'];
            const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();

            if (!validTypes.includes(ext)) {
                alert('仅支持 Word、PDF、Excel 和 CSV 文件');
                return;
            }

            if (file.size > 10 * 1024 * 1024) {
                alert('文件过大，请限制在 10MB 以内');
                return;
            }

            // 检查是否已存在
            if (!selectedFiles.some(f => f.name === file.name && f.size === file.size)) {
                selectedFiles.push(file);
            }

            updateFileDisplay();
        }

        function updateFileDisplay() {
            if (selectedFiles.length === 0) {
                selectedFileDiv.style.display = 'none';
                uploadSubmitBtn.disabled = true;
                return;
            }

            if (selectedFiles.length === 1) {
                const file = selectedFiles[0];
                const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
                document.getElementById('fileIcon').textContent = getFileIcon(ext);
                fileName.textContent = file.name;
                fileSize.textContent = (file.size / 1024).toFixed(1) + ' KB';
                selectedFileDiv.style.display = 'flex';
            } else {
                document.getElementById('fileIcon').textContent = '📁';
                fileName.textContent = `${selectedFiles.length} 个文件`;
                const totalSizeKb = (selectedFiles.reduce((total, file) => total + file.size, 0) / 1024).toFixed(1);
                fileSize.textContent = `${totalSizeKb} KB`;
                selectedFileDiv.style.display = 'flex';
            }
            uploadSubmitBtn.disabled = false;
        }

        function clearFiles() {
            selectedFiles = [];
            fileInput.value = '';
            selectedFileDiv.style.display = 'none';
            uploadSubmitBtn.disabled = true;
        }

        function removeFile(index) {
            selectedFiles.splice(index, 1);
            updateFileDisplay();
        }

        async function startUpload() {
            if (selectedFiles.length === 0) return;

            uploadSubmitBtn.disabled = true;
            uploadSubmitBtn.innerHTML = '<div class="loading-spinner"></div> 上传中...';
            let successCount = 0;
            let duplicateCount = 0;
            let failureCount = 0;
            const failureMessages = [];

            try {
                // 逐个上传文件
                for (let i = 0; i < selectedFiles.length; i++) {
                    const file = selectedFiles[i];
                    const formData = new FormData();
                    formData.append('file', file);
                    formData.append('mode', currentUploadMode);

                    if (currentUploadMode === 'knowledge') {
                        const categoryVal = document.getElementById('categorySelect').value;
                        if (!categoryVal) {
                            alert('请选择知识库分类');
                            uploadSubmitBtn.disabled = false;
                            uploadSubmitBtn.textContent = '开始上传';
                            return;
                        }
                        const catInfo = JSON.parse(categoryVal);
                        formData.append('category', catInfo.category);
                        formData.append('department', catInfo.department);
                    }

                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });

                    let data = {};
                    const rawText = await response.text();
                    try {
                        data = rawText ? JSON.parse(rawText) : {};
                    } catch (parseError) {
                        data = {
                            success: false,
                            message: response.ok ? '上传响应解析失败' : `上传失败（HTTP ${response.status}）`
                        };
                    }

                    if (!response.ok) {
                        if (response.status === 401) {
                            data.message = '登录已过期，请重新登录';
                            setTimeout(() => { localStorage.removeItem('token'); location.href = '/login'; }, 1200);
                        } else if (response.status === 403) {
                            data.message = data.message || '没有上传到该知识库分类的权限';
                        } else if (response.status === 413) {
                            data.message = '文件过大，请压缩后重试';
                        } else {
                            data.message = data.message || data.error || `上传失败（HTTP ${response.status}）`;
                        }
                        data.success = false;
                    }

                    if (data.success && currentUploadMode === 'knowledge' && data.job_id) {
                        uploadSubmitBtn.innerHTML = '<div class="loading-spinner"></div> 入库中...';
                        const job = await waitForUploadJob(data.job_id);
                        data = job.result || {};
                        if (job.status === 'failed') {
                            data.success = false;
                            data.message = job.error || job.message || data.message || '入库失败';
                        } else if (data.success !== false) {
                            data.success = true;
                        }
                    }

                    if (data.success) {
                        successCount += 1;

                        // 临时上传的文件保存到附件列表
                        if (currentUploadMode === 'temp' && data.file_id) {
                            const uploadedFile = {
                                file_id: data.file_id,
                                filename: data.filename,
                                char_count: data.char_count,
                                is_spreadsheet: isSpreadsheetFilename(data.filename)
                            };
                            if (!attachedFiles.some(item => item.file_id === uploadedFile.file_id)) {
                                attachedFiles.push(uploadedFile);
                            }
                            updateAttachedFilesDisplay();
                        }
                    } else {
                        if (data.duplicate) {
                            duplicateCount += 1;
                        } else {
                            failureCount += 1;
                            if (data.message || data.error) {
                                failureMessages.push(data.message || data.error);
                            }
                        }
                    }
                }

                hideUploadModal();
                const summaryParts = [`成功 ${successCount}`];
                if (duplicateCount) summaryParts.push(`重复 ${duplicateCount}`);
                if (failureCount) summaryParts.push(`失败 ${failureCount}`);
                if (currentUploadMode === 'temp' && successCount > 0) {
                    showInlineNotice(`已添加到输入框：${summaryParts.join('，')}`);
                } else {
                    const detail = failureMessages.length ? `；${failureMessages[0]}` : '';
                    showInlineNotice(`上传处理完成：${summaryParts.join('，')}${detail}`);
                }
                clearFiles(); // 清空已选文件
            } catch (error) {
                console.error('上传失败:', error);
                showInlineNotice('上传失败，请重试');
            } finally {
                uploadSubmitBtn.disabled = false;
                uploadSubmitBtn.textContent = '开始上传';
                refreshComposerState();
            }
        }

        // 更新已上传文件显示
        function updateAttachedFilesDisplay() {
            const oldDisplay = document.getElementById('attachedFilesDisplay');
            if (!oldDisplay) return;

            oldDisplay.hidden = attachedFiles.length === 0;
            if (attachedFiles.length === 0) {
                oldDisplay.innerHTML = '';
                refreshEditorSources();
                refreshComposerState();
                return;
            }

            oldDisplay.innerHTML = `
                ${attachedFiles.map((file, index) => `
                    <span class="attached-file-chip" title="${escapeHtml(file.filename || '附件')}">
                        <span class="attached-file-name">${escapeHtml(file.filename || '附件')}</span>
                        ${Number.isFinite(Number(file.char_count)) ? `<span class="attached-file-meta">${Number(file.char_count)}字</span>` : ''}
                        ${(file.is_spreadsheet || isSpreadsheetFilename(file.filename)) ? `<button type="button" class="attached-file-action" onclick="showSpreadsheetTransformModal(${index})" title="处理并导出表格">处理</button>` : ''}
                        <button type="button" class="attached-file-remove" onclick="removeAttachedFile(${index})" title="移除附件">移除</button>
                    </span>
                `).join('')}
                <button type="button" class="attached-files-clear" onclick="clearAttachedFiles()" title="清除全部附件">清除</button>
            `;
            refreshEditorSources();
            refreshComposerState();
        }

        // 移除单个附加文件
        function removeAttachedFile(index) {
            attachedFiles.splice(index, 1);
            updateAttachedFilesDisplay();
        }

        // 清除所有附加文件
        function clearAttachedFiles() {
            attachedFiles = [];
            updateAttachedFilesDisplay();
        }

        function showSpreadsheetTransformModal(index) {
            const file = attachedFiles[index];
            if (!file) {
                showInlineNotice('未找到该表格附件');
                return;
            }
            spreadsheetTransformFile = file;
            document.getElementById('spreadsheetTransformFilename').value = file.filename || '';
            spreadsheetTransformInstruction.value = '';
            spreadsheetTransformModal.classList.add('show');
            setTimeout(() => spreadsheetTransformInstruction?.focus(), 40);
        }

        function hideSpreadsheetTransformModal() {
            spreadsheetTransformModal?.classList.remove('show');
            spreadsheetTransformFile = null;
        }

        function submitSpreadsheetTransform() {
            const instruction = spreadsheetTransformInstruction.value.trim();
            if (!spreadsheetTransformFile) {
                showInlineNotice('请先选择表格附件');
                return;
            }
            if (!instruction) {
                showInlineNotice('请填写处理规则');
                spreadsheetTransformInstruction.focus();
                return;
            }

            const existingIndex = attachedFiles.findIndex(file => file.file_id === spreadsheetTransformFile.file_id);
            if (existingIndex < 0) {
                attachedFiles.push(spreadsheetTransformFile);
            }
            hideSpreadsheetTransformModal();
            chatInput.value = instruction;
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
            refreshComposerState();
            sendMessage();
        }

        // 点击弹窗外部关闭
        uploadModal.addEventListener('click', (e) => {
            if (e.target === uploadModal) hideUploadModal();
        });

        feedbackModal?.addEventListener('click', (e) => {
            if (e.target === feedbackModal) hideFeedbackModal();
        });

        spreadsheetTransformModal?.addEventListener('click', (e) => {
            if (e.target === spreadsheetTransformModal) hideSpreadsheetTransformModal();
        });

        document.addEventListener('click', (e) => {
            if (window.innerWidth > 1024) return;
            const sidebar = document.querySelector('.sidebar-section');
            const toggle = document.querySelector('.mobile-nav-btn');
            if (!document.body.classList.contains('sidebar-open')) return;
            if (sidebar && sidebar.contains(e.target)) return;
            if (toggle && toggle.contains(e.target)) return;
            document.body.classList.remove('sidebar-open');
        });

        // 优化文件输入，支持多文件选择
        fileInput.setAttribute('multiple', 'multiple');

        // ==================== 退出登录 ====================
        async function logout() {
            if (confirm('确定要退出登录吗？')) {
                // 调用后端清除 cookie
                await fetch('/api/auth/logout', { method: 'POST' });
                localStorage.removeItem('token');
                localStorage.removeItem('user');
                location.href = '/';
            }
        }

        // 初始化：编辑器入口常驻，方便内测用户随时打开空白文档。
        editorToggle.classList.remove('hidden');

        // ==================== 网络状态检测 ====================
        const networkStatus = document.getElementById('networkStatus');
        let offlineTimer = null;

        function showNetworkStatus(type) {
            if (!networkStatus) return;
            networkStatus.classList.add('show');
            networkStatus.classList.toggle('restored', type === 'restored');
            networkStatus.innerHTML = type === 'offline'
                ? '<span>⚠️ 网络连接已断开，请检查网络</span>'
                : '<span>✅ 网络已恢复</span>';

            if (type === 'restored') {
                clearTimeout(offlineTimer);
                offlineTimer = setTimeout(() => {
                    networkStatus.classList.remove('show', 'restored');
                }, 3000);
            }
        }

        window.addEventListener('online', () => showNetworkStatus('restored'));
        window.addEventListener('offline', () => showNetworkStatus('offline'));

        // 全局 fetch 错误监听，自动提示
        const originalFetch = window.fetch;
        window.fetch = function(...args) {
            return originalFetch.apply(this, args).then(response => {
                if (response.status === 429) {
                    showNetworkStatus('offline');
                    networkStatus.innerHTML = '<span>⚠️ 请求过于频繁，请稍后重试</span>';
                    setTimeout(() => networkStatus.classList.remove('show'), 5000);
                }
                return response;
            }).catch(err => {
                if (!navigator.onLine) {
                    showNetworkStatus('offline');
                }
                throw err;
            });
        };
