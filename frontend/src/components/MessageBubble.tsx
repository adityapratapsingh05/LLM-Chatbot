import { ChatMetadata } from '../types';
import styles from './MessageBubble.module.css';

interface AssistantBubbleProps {
  content: string;
  metadata: ChatMetadata;
}

function MetaBadge({ source }: { source: 'LIVE_INFERENCE' | 'LLM_CACHE' }) {
  const isCache = source === 'LLM_CACHE';
  return (
    <span
      data-testid="meta-source"
      className={`${styles.badge} ${isCache ? styles.badgeCache : styles.badgeLive}`}
    >
      {source}
    </span>
  );
}

export function AssistantBubble({ content, metadata }: AssistantBubbleProps) {
  return (
    <div data-testid="assistant-message" className={styles.assistantBubble}>
      <div className={styles.bubbleContent}>
        <span className={styles.roleIcon}>🤖</span>
        <div className={styles.textBlock}>
          <p className={styles.messageText}>{content}</p>
          <div className={styles.meta}>
            <MetaBadge source={metadata.resolution_source} />
            <span className={styles.metaItem}>
              <span className={styles.metaLabel}>Model:</span>
              <span data-testid="meta-model" className={styles.metaValue}>{metadata.model}</span>
            </span>
            <span className={styles.metaItem}>
              <span className={styles.metaLabel}>Latency:</span>
              <span data-testid="meta-latency" className={styles.metaValue}>{metadata.latency_ms} ms</span>
            </span>
            <span className={styles.metaItem}>
              <span className={styles.metaLabel}>In:</span>
              <span data-testid="meta-input-tokens" className={styles.metaValue}>{metadata.input_tokens}</span>
              <span className={styles.metaLabel}> Out:</span>
              <span data-testid="meta-output-tokens" className={styles.metaValue}>{metadata.output_tokens}</span>
              <span className={styles.metaLabel}> tok</span>
            </span>
            {metadata.key_id && (
              <span className={styles.metaItem}>
                <span className={styles.metaLabel}>Key:</span>
                <span className={styles.metaValue}>{metadata.key_id}</span>
              </span>
            )}
            {metadata.fallback_attempts > 0 && (
              <span className={`${styles.metaItem} ${styles.fallback}`}>
                ⚡ {metadata.fallback_attempts} fallback{metadata.fallback_attempts > 1 ? 's' : ''}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

interface UserBubbleProps {
  content: string;
}

export function UserBubble({ content }: UserBubbleProps) {
  return (
    <div className={styles.userBubble}>
      <div className={styles.bubbleContent}>
        <div className={styles.textBlock}>
          <p className={styles.messageText}>{content}</p>
        </div>
        <span className={styles.roleIcon}>👤</span>
      </div>
    </div>
  );
}

interface ErrorBubbleProps {
  content: string;
}

export function ErrorBubble({ content }: ErrorBubbleProps) {
  return (
    <div className={styles.errorBubble}>
      <span>⚠️ {content}</span>
    </div>
  );
}
