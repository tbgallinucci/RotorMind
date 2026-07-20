document.addEventListener('DOMContentLoaded', () => {
    const wikiNav = document.getElementById('wiki-nav');
    const wikiSearch = document.getElementById('wiki-search');
    const wikiContent = document.getElementById('wiki-content');
    const pageTitle = document.getElementById('page-title');
    const chatHistory = document.getElementById('chat-history');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const clearChat = document.getElementById('clear-chat');
    const toggleSidebar = document.getElementById('toggle-sidebar');
    const downloadRaw = document.getElementById('download-raw');
    const leftSidebar = document.getElementById('sidebar');
    const rightSidebar = document.getElementById('chat-sidebar');
    const leftResizer = document.getElementById('left-resizer');
    const rightResizer = document.getElementById('right-resizer');
    const pageSearch = document.getElementById('page-search');
    const pageSearchCount = document.getElementById('page-search-count');
    const pageSearchPrev = document.getElementById('page-search-prev');
    const pageSearchNext = document.getElementById('page-search-next');
    const modeLocalBtn = document.getElementById('mode-local');
    const modeCloudBtn = document.getElementById('mode-cloud');

    let treeData = [];
    let history = [];
    let activePage = '';
    let currentMatches = [];
    let currentMatchIndex = -1;
    let llmMode = localStorage.getItem('rotormind-llm-mode') || 'local';

    // Initialize Marked.js
    marked.setOptions({
        gfm: true,
        breaks: true,
        headerIds: true,
        mangle: false
    });

    // Render LaTeX (KaTeX auto-render) inside an element; no-op if CDN unavailable
    function renderMath(el) {
        if (!window.renderMathInElement || !el) return;
        try {
            renderMathInElement(el, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '\\[', right: '\\]', display: true },
                    { left: '\\(', right: '\\)', display: false },
                    { left: '$', right: '$', display: false },
                ],
                throwOnError: false,
            });
        } catch (e) { /* leave raw text */ }
    }

    // Load Wiki Tree
    async function loadPages() {
        try {
            const response = await fetch('/api/tree');
            treeData = await response.json();
            renderWikiTree(treeData, wikiNav);
        } catch (error) {
            console.error('Error loading tree:', error);
            wikiNav.innerHTML = '<div class="error">Failed to load wiki index</div>';
        }
    }

    function renderWikiTree(nodes, container) {
        container.innerHTML = '';
        nodes.forEach(node => {
            const wrapper = document.createElement('div');
            wrapper.className = 'wiki-tree-item';

            if (node.type === 'directory' || node.type === 'pdf') {
                const isPdf = node.type === 'pdf';
                const folder = document.createElement('div');
                
                // Check if this folder contains the active page or an active child
                const hasActiveChild = checkHasActiveChild(node, activePage);
                
                folder.className = `wiki-folder ${isPdf ? 'pdf-node' : ''} ${hasActiveChild ? '' : 'collapsed'}`;
                folder.innerHTML = `
                    <i class="fas fa-chevron-down"></i> 
                    <i class="fas ${isPdf ? 'fa-file-pdf' : 'fa-folder'}"></i> 
                    <span class="node-name">${node.name}</span>
                `;
                
                const childrenContainer = document.createElement('div');
                childrenContainer.className = `wiki-folder-children ${hasActiveChild ? '' : 'collapsed'}`;
                
                folder.onclick = (e) => {
                    e.stopPropagation();
                    folder.classList.toggle('collapsed');
                    childrenContainer.classList.toggle('collapsed');
                };

                renderWikiTree(node.children, childrenContainer);
                
                wrapper.appendChild(folder);
                wrapper.appendChild(childrenContainer);
            } else {
                const item = document.createElement('div');
                item.className = `wiki-item ${activePage === node.slug ? 'active' : ''}`;
                item.innerHTML = `
                    <span class="slug">${node.slug}</span>
                    <span class="desc">${node.description || ''}</span>
                    ${node.snippet ? `<div class="search-snippet">${node.snippet}</div>` : ''}
                `;
                item.onclick = () => loadPage(node.slug);
                wrapper.appendChild(item);
            }
            container.appendChild(wrapper);
        });
    }

    function checkHasActiveChild(node, activeSlug) {
        if (!activeSlug) return false;
        if (node.children) {
            return node.children.some(child => {
                if (child.type === 'file') {
                    return child.slug === activeSlug;
                }
                return checkHasActiveChild(child, activeSlug);
            });
        }
        return false;
    }

    // Full-text search with debounce
    let searchTimeout;
    wikiSearch.oninput = (e) => {
        clearTimeout(searchTimeout);
        const query = e.target.value.trim();
        
        if (!query) {
            renderWikiTree(treeData, wikiNav);
            return;
        }

        searchTimeout = setTimeout(async () => {
            try {
                const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
                const results = await response.json();
                renderWikiTree(results, wikiNav);
            } catch (error) {
                console.error('Search error:', error);
            }
        }, 300);
    };

    // Load Specific Page
    async function loadPage(slug) {
        window.dispatchEvent(new Event('copilot:show-wiki'));
        activePage = slug;
        renderWikiTree(treeData, wikiNav);
        
        try {
            wikiContent.innerHTML = '<div class="loading">Loading content...</div>';
            const response = await fetch(`/api/pages/${slug}`);
            const data = await response.json();
            
            pageTitle.innerText = slug.split('/').pop();
            wikiContent.innerHTML = marked.parse(data.content);
            renderMath(wikiContent);
            
            // Show download button
            downloadRaw.style.display = 'block';
            downloadRaw.title = "Download Source PDF";
            downloadRaw.onclick = () => {
                window.location.href = `/api/download-pdf/${slug}`;
            };
            
            // Intercept links in the rendered markdown
            interceptLinks();
            
            // Highlight active in sidebar
            renderWikiTree(treeData, wikiNav);
            
            // Reset page search
            pageSearch.value = '';
            pageSearchCount.innerText = '';
            currentMatches = [];
            currentMatchIndex = -1;
            
            // Scroll to top
            wikiContent.scrollTop = 0;
        } catch (error) {
            console.error('Error loading page:', error);
            wikiContent.innerHTML = `<div class="error">Error loading page ${slug}</div>`;
            downloadRaw.style.display = 'none';
        }
    }

    // Link Interception Logic
    function interceptLinks() {
        // Intercept [[page-name]] links (which marked.js might render as text or links)
        // And standard markdown links
        const links = wikiContent.querySelectorAll('a');
        links.forEach(link => {
            const href = link.getAttribute('href');
            if (href && !href.startsWith('http')) {
                link.onclick = (e) => {
                    e.preventDefault();
                    // Clean slug (remove .md, leading/trailing slashes)
                    const slug = href.replace('.md', '').replace(/^\/+|\/+$/g, '');
                    loadPage(slug);
                };
            }
        });

        // Also search for [[wiki-link]] patterns in text that might not have been converted
        const textNodes = [];
        const walk = document.createTreeWalker(wikiContent, NodeFilter.SHOW_TEXT, null, false);
        let node;
        while(node = walk.nextNode()) textNodes.push(node);

        textNodes.forEach(node => {
            const text = node.nodeValue;
            if (text.includes('[[')) {
                const span = document.createElement('span');
                span.innerHTML = text.replace(/\[\[([^\]]+)\]\]/g, (match, slug) => {
                    return `<a href="#" class="wiki-link-dynamic" data-slug="${slug}">[[${slug}]]</a>`;
                });
                node.parentNode.replaceChild(span, node);
            }
        });

        wikiContent.querySelectorAll('.wiki-link-dynamic').forEach(link => {
            link.onclick = (e) => {
                e.preventDefault();
                loadPage(link.getAttribute('data-slug'));
            };
        });
    }

    // Local/Cloud LLM toggle
    function setLlmMode(mode) {
        llmMode = mode;
        localStorage.setItem('rotormind-llm-mode', mode);
        modeLocalBtn.classList.toggle('active', mode === 'local');
        modeCloudBtn.classList.toggle('active', mode === 'cloud');
    }

    async function initLlmModeToggle() {
        modeLocalBtn.onclick = () => setLlmMode('local');
        modeCloudBtn.onclick = () => setLlmMode('cloud');
        try {
            const response = await fetch('/api/llm-status');
            const status = await response.json();
            if (!status.cloud) {
                modeCloudBtn.disabled = true;
                modeCloudBtn.title = 'Cloud mode unavailable: no CLOUD_LLM_API_KEY configured on the server';
                if (llmMode === 'cloud') llmMode = 'local';
            } else {
                modeCloudBtn.title = `Cloud mode: ${status.cloud_model}`;
            }
            if (!status.local) {
                modeLocalBtn.disabled = true;
                modeLocalBtn.title = 'Local mode unavailable: LM Studio client not configured';
                if (llmMode === 'local') llmMode = 'cloud';
            }
        } catch (error) {
            console.error('Could not fetch LLM status:', error);
        }
        setLlmMode(llmMode);
    }

    // Chat Logic
    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        chatInput.value = '';
        chatInput.style.height = 'auto';

        // Add user message to UI
        addChatMessage('user', message);

        // Add assistant placeholder
        const assistantMsgDiv = addChatMessage('assistant', '');
        const contentDiv = assistantMsgDiv.querySelector('.message-content');
        contentDiv.innerHTML = '<span class="typing-indicator">Thinking...</span>';

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, history, mode: llmMode })
            });

            if (!response.ok) throw new Error('Chat API failed');

            contentDiv.innerHTML = '';
            assistantMsgDiv.classList.add('typing');
            let displayed = '';   // what the user currently sees
            let lineBuffer = '';  // partial JSON line split across chunk boundaries

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                lineBuffer += decoder.decode(value, { stream: true });
                const lines = lineBuffer.split('\n');
                lineBuffer = lines.pop(); // last (possibly partial) line stays buffered

                for (const line of lines) {
                    if (!line.trim()) continue;
                    let evt;
                    try { evt = JSON.parse(line); } catch { continue; }

                    if (evt.type === 'delta') {
                        // Optimistic: token arrives, shown immediately.
                        displayed += evt.text;
                        contentDiv.innerHTML = parseAssistantResponse(displayed) + '<span class="cursor"></span>';
                        chatHistory.scrollTop = chatHistory.scrollHeight;
                    } else if (evt.type === 'replace') {
                        // Clears an abandoned tool-call preamble (the model
                        // thought out loud, then called a tool instead of
                        // answering) — text is empty, more deltas follow.
                        displayed = evt.text || '';
                        contentDiv.innerHTML = parseAssistantResponse(displayed) + '<span class="cursor"></span>';
                        chatHistory.scrollTop = chatHistory.scrollHeight;
                    } else if (evt.type === 'flag') {
                        // Grounding guard: a claim couldn't be verified against
                        // its cited source. The answer text is left exactly as
                        // streamed — we only append a warning note below it.
                        let flagDiv = assistantMsgDiv.querySelector('.grounding-flag');
                        if (!flagDiv) {
                            flagDiv = document.createElement('div');
                            flagDiv.className = 'grounding-flag';
                            assistantMsgDiv.appendChild(flagDiv);
                        }
                        flagDiv.innerHTML = '<i class="fas fa-triangle-exclamation"></i> '
                            + 'Could not verify against the cited source: '
                            + escapeHtml(evt.text || '');
                        chatHistory.scrollTop = chatHistory.scrollHeight;
                    } else if (evt.type === 'error') {
                        displayed = evt.text || 'Error communicating with AI assistant.';
                        contentDiv.innerHTML = escapeHtml(displayed);
                    }
                    // 'done' needs no action — displayed already holds the final text.
                }
            }

            // Remove cursor and typing class
            assistantMsgDiv.classList.remove('typing');
            contentDiv.innerHTML = parseAssistantResponse(displayed);
            renderMath(contentDiv);

            // Update history
            history.push({ role: 'user', content: message });
            history.push({ role: 'assistant', content: displayed });

            // The reply may have ingested a new run report - refresh the tree
            loadPages();

        } catch (error) {
            console.error('Chat error:', error);
            contentDiv.innerHTML = 'Error communicating with AI assistant. Make sure LM Studio is running.';
        }
    }

    function addChatMessage(role, content) {
        const div = document.createElement('div');
        div.className = `chat-message ${role}`;
        div.innerHTML = `
            <div class="message-content">
                ${role === 'assistant' ? parseAssistantResponse(content) : escapeHtml(content)}
            </div>
        `;
        chatHistory.appendChild(div);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return div;
    }

    function parseAssistantResponse(text) {
        const links = [];
        
        // 1. Extract different citation patterns and replace with safe placeholders
        // Pattern A: wiki: [[slug]] or [[slug]]
        let processed = text.replace(/(?:wiki:\s*)?\[\[([^\]]+)\]\]/gi, (match, slug) => {
            const cleanSlug = slug.trim();
            const id = links.length;
            links.push(`(<span class="citation" onclick="window.loadWikiPage('${cleanSlug}')">wiki: ${cleanSlug}</span>)`);
            return `WIKILINKVIBRANT${id}`;
        });

        // Pattern R: (run: page-id) -> opens the ingested run report page
        processed = processed.replace(/[\[\(]run:\s*([^,\]\)]+)[\]\)]/gi, (match, slug) => {
            const cleanSlug = slug.trim();
            const id = links.length;
            links.push(`(<span class="citation" onclick="window.loadWikiPage('${cleanSlug}')">run: ${cleanSlug}</span>)`);
            return `WIKILINKVIBRANT${id}`;
        });

        // Pattern B: (wiki: slug, section) or [wiki: slug, section]
        processed = processed.replace(/[\[\(]wiki:\s*([^,\]\)]+)(?:,\s*([^\]\)]+))?[\]\)]/gi, (match, slug, section) => {
            const cleanSlug = slug.trim();
            const display = section ? `${cleanSlug} > ${section.trim()}` : cleanSlug;
            const id = links.length;
            links.push(`(<span class="citation" onclick="window.loadWikiPage('${cleanSlug}')">wiki: ${display}</span>)`);
            return `WIKILINKVIBRANT${id}`;
        });

        // 2. Render Markdown
        let html = marked.parse(processed);

        // 3. Inject the real links back into the HTML
        links.forEach((linkHtml, i) => {
            // Using split/join for global replacement without regex escaping issues
            html = html.split(`WIKILINKVIBRANT${i}`).join(linkHtml);
        });

        return html;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Global helper for citation clicks
    window.loadWikiPage = (slug) => {
        loadPage(slug);
    };

    // UI Events
    sendButton.onclick = sendMessage;
    chatInput.onkeydown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
        // Auto-expand textarea
        chatInput.style.height = 'auto';
        chatInput.style.height = (chatInput.scrollHeight) + 'px';
    };

    clearChat.onclick = () => {
        history = [];
        chatHistory.innerHTML = '<div class="chat-message assistant"><div class="message-content">History cleared. How can I help you today?</div></div>';
    };

    // Resizer Logic
    function initResizer(resizer, element, side) {
        let x = 0;
        let w = 0;

        const mouseDownHandler = function(e) {
            x = e.clientX;
            const styles = window.getComputedStyle(element);
            w = parseInt(styles.width, 10);

            document.addEventListener('mousemove', mouseMoveHandler);
            document.addEventListener('mouseup', mouseUpHandler);
            resizer.classList.add('resizing');
        };

        const mouseMoveHandler = function(e) {
            const dx = e.clientX - x;
            if (side === 'left') {
                element.style.width = `${w + dx}px`;
            } else {
                element.style.width = `${w - dx}px`;
            }
        };

        const mouseUpHandler = function() {
            document.removeEventListener('mousemove', mouseMoveHandler);
            document.removeEventListener('mouseup', mouseUpHandler);
            resizer.classList.remove('resizing');
        };

        resizer.addEventListener('mousedown', mouseDownHandler);
    }

    initResizer(leftResizer, leftSidebar, 'left');
    initResizer(rightResizer, rightSidebar, 'right');

    // Page Search Logic
    function highlightSearchInPage(query) {
        // Remove existing highlights
        const marks = wikiContent.querySelectorAll('mark.search-highlight');
        marks.forEach(mark => {
            const text = document.createTextNode(mark.innerText);
            mark.parentNode.replaceChild(text, mark);
        });
        
        wikiContent.normalize(); // Merge adjacent text nodes

        if (!query) {
            pageSearchCount.innerText = '';
            currentMatches = [];
            currentMatchIndex = -1;
            return;
        }

        const walker = document.createTreeWalker(wikiContent, NodeFilter.SHOW_TEXT, null, false);
        const nodes = [];
        let node;
        while (node = walker.nextNode()) nodes.push(node);

        currentMatches = [];
        const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');

        nodes.forEach(textNode => {
            const matches = textNode.nodeValue.match(regex);
            if (matches) {
                const span = document.createElement('span');
                span.innerHTML = textNode.nodeValue.replace(regex, '<mark class="search-highlight">$1</mark>');
                
                const newMarks = Array.from(span.querySelectorAll('mark'));
                currentMatches.push(...newMarks);
                
                textNode.parentNode.replaceChild(span, textNode);
                // Unwrapping the span but keeping the children
                while (span.firstChild) {
                    span.parentNode.insertBefore(span.firstChild, span);
                }
                span.parentNode.removeChild(span);
            }
        });

        if (currentMatches.length > 0) {
            currentMatchIndex = 0;
            updateMatchNavigation();
        } else {
            pageSearchCount.innerText = '0/0';
            currentMatchIndex = -1;
        }
    }

    function updateMatchNavigation() {
        if (currentMatches.length === 0) return;

        // Remove active class from all
        currentMatches.forEach(m => m.classList.remove('active'));

        // Add to current
        const current = currentMatches[currentMatchIndex];
        current.classList.add('active');
        current.scrollIntoView({ behavior: 'smooth', block: 'center' });

        pageSearchCount.innerText = `${currentMatchIndex + 1}/${currentMatches.length}`;
    }

    pageSearch.oninput = (e) => {
        highlightSearchInPage(e.target.value);
    };

    pageSearch.onkeydown = (e) => {
        if (e.key === 'Enter') {
            if (e.shiftKey) {
                pageSearchPrev.click();
            } else {
                pageSearchNext.click();
            }
        }
    };

    pageSearchPrev.onclick = () => {
        if (currentMatches.length === 0) return;
        currentMatchIndex = (currentMatchIndex - 1 + currentMatches.length) % currentMatches.length;
        updateMatchNavigation();
    };

    pageSearchNext.onclick = () => {
        if (currentMatches.length === 0) return;
        currentMatchIndex = (currentMatchIndex + 1) % currentMatches.length;
        updateMatchNavigation();
    };

    // UI Events
    toggleSidebar.onclick = () => {
        leftSidebar.classList.toggle('collapsed');
    };

    // Light/dark theme toggle (persisted)
    const themeToggle = document.getElementById('theme-toggle');
    document.documentElement.dataset.theme = localStorage.getItem('rotormind-theme') || 'dark';
    function syncThemeIcon() {
        themeToggle.innerHTML = document.documentElement.dataset.theme === 'dark'
            ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
    }
    themeToggle.onclick = () => {
        const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
        document.documentElement.dataset.theme = next;
        localStorage.setItem('rotormind-theme', next);
        syncThemeIcon();
    };
    syncThemeIcon();

    loadPages();
    initLlmModeToggle();

    // Expose for the FEA run panel (run-panel.js)
    window.copilotUI = { loadPages, loadPage };
});
