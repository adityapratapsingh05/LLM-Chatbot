import { useState, useRef, useEffect, useCallback } from 'react';
import { Message } from './types';
import { sendChat } from './api';
import { AssistantBubble, UserBubble, ErrorBubble } from './components/MessageBubble';
import styles from './App.module.css';

function generateSessionId(): string {
  return 's-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}

const MAX_LENGTH = 4000;

export default function App() {
  const [sessionId] = useState(() => generateSessionId());
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const handleSend = useCallback(async () => {
    const trimmed = inputValue.trim();
    if (!trimmed || isLoading) return;
    if (trimmed.length > MAX_LENGTH) return;

    const userMessage: Message = {
      id: `msg-${Date.now()}-user`,
      role: 'user',
      content: trimmed,
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await sendChat(sessionId, trimmed, systemPrompt || undefined);
      const assistantMessage: Message = {
        id: `msg-${Date.now()}-assistant`,
        role: 'assistant',
        content: response.reply,
        metadata: response.metadata,
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An unexpected error occurred.';
      let friendlyMsg = errorMsg;
      if (errorMsg.toLowerCase().includes('all providers unavailable')) {
        friendlyMsg = 'All providers are unavailable, please try again.';
      }
      const errorMessage: Message = {
        id: `msg-${Date.now()}-error`,
        role: 'error',
        content: friendlyMsg,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [inputValue, isLoading, sessionId, systemPrompt]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const handleNewChat = () => {
    window.location.reload();
  };

  const charCount = inputValue.length;
  const isOverLimit = charCount > MAX_LENGTH;
  const canSend = inputValue.trim().length > 0 && !isLoading && !isOverLimit;

  return (
    <div className={styles.app}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <div className={styles.logo}>⚡ LLM Chat</div>
          <span className={styles.sessionTag}>Session: {sessionId}</span>
        </div>
        <button className={styles.newChatBtn} onClick={handleNewChat} title="Start new chat">
          + New Chat
        </button>
      </header>

      {/* System prompt */}
      <div className={styles.systemPromptBar}>
        <label className={styles.systemPromptLabel} htmlFor="system-prompt-input">
          System Prompt
        </label>
        <input
          id="system-prompt-input"
          data-testid="system-prompt-input"
          type="text"
          className={styles.systemPromptInput}
          placeholder="You are a helpful assistant. (optional)"
          value={systemPrompt}
          onChange={e => setSystemPrompt(e.target.value)}
        />
      </div>

      {/* Messages */}
      <main className={styles.messageArea}>
        {messages.length === 0 && (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>💬</div>
            <p>Start a conversation! Type a message below.</p>
          </div>
        )}
        {messages.map(msg => {
          if (msg.role === 'user') return <UserBubble key={msg.id} content={msg.content} />;
          if (msg.role === 'assistant' && msg.metadata)
            return <AssistantBubble key={msg.id} content={msg.content} metadata={msg.metadata} />;
          if (msg.role === 'error') return <ErrorBubble key={msg.id} content={msg.content} />;
          return null;
        })}
        {isLoading && (
          <div className={styles.typingIndicator}>
            <span></span><span></span><span></span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      {/* Input area */}
      <footer className={styles.inputArea}>
        <div className={styles.inputWrapper}>
          <textarea
            ref={inputRef}
            data-testid="chat-input"
            className={`${styles.chatInput} ${isOverLimit ? styles.chatInputError : ''}`}
            placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            rows={1}
            maxLength={MAX_LENGTH + 100} // allow typing past, we validate
          />
          <div className={styles.inputFooter}>
            <span className={`${styles.charCount} ${isOverLimit ? styles.charCountError : ''}`}>
              {charCount}/{MAX_LENGTH}
            </span>
            <button
              data-testid="send-button"
              className={styles.sendButton}
              onClick={handleSend}
              disabled={!canSend}
              aria-label="Send message"
            >
              {isLoading ? (
                <span className={styles.loadingSpinner} />
              ) : (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
