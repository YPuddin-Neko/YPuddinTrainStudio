import { useEffect, useRef } from 'react';
import { EventType, EVENT_TYPES } from './eventTypes';

type EventCallback<T = any> = (data: T) => void;

class EventStreamManager {
  private static instance: EventStreamManager;
  private eventSource: EventSource | null = null;
  private listeners: Map<string, Set<EventCallback>> = new Map();
  private lastEventId: string = '';
  private reconnectTimer: any = null;
  private isConnecting: boolean = false;

  private constructor() {}

  public static getInstance(): EventStreamManager {
    if (!EventStreamManager.instance) {
      EventStreamManager.instance = new EventStreamManager();
    }
    return EventStreamManager.instance;
  }

  public subscribe<T>(type: EventType | '*', callback: EventCallback<T>): () => void {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, new Set());
    }
    this.listeners.get(type)!.add(callback);

    this.ensureConnection();

    return () => {
      const set = this.listeners.get(type);
      if (set) {
        set.delete(callback);
        if (set.size === 0) {
          this.listeners.delete(type);
        }
      }
    };
  }

  private ensureConnection() {
    if (this.eventSource || this.isConnecting || typeof window === 'undefined') return;

    this.isConnecting = true;
    const url = new URL('/api/events', window.location.origin);
    if (this.lastEventId) {
      url.searchParams.set('last_event_id', this.lastEventId);
    }

    try {
      this.eventSource = new EventSource(url.toString());

      this.eventSource.onopen = () => {
        this.isConnecting = false;
        if (this.reconnectTimer) {
          clearTimeout(this.reconnectTimer);
          this.reconnectTimer = null;
        }
      };

      // 监听所有预定义的事件类型
      Object.values(EVENT_TYPES).forEach((eventType) => {
        this.eventSource?.addEventListener(eventType, (e: MessageEvent) => {
          if (e.lastEventId) {
            this.lastEventId = e.lastEventId;
          }
          try {
            const data = JSON.parse(e.data);
            this.dispatch(eventType, data);
          } catch (err) {
            console.error(`Failed to parse SSE event data for ${eventType}:`, err);
          }
        });
      });

      this.eventSource.onerror = () => {
        this.isConnecting = false;
        this.eventSource?.close();
        this.eventSource = null;
        // 断线 3 秒重连
        if (!this.reconnectTimer) {
          this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.ensureConnection();
          }, 3000);
        }
      };
    } catch (e) {
      this.isConnecting = false;
    }
  }

  private dispatch(type: string, data: any) {
    const specific = this.listeners.get(type);
    if (specific) {
      specific.forEach((cb) => {
        try {
          cb(data);
        } catch (e) {
          console.error(e);
        }
      });
    }

    const wildcard = this.listeners.get('*');
    if (wildcard) {
      wildcard.forEach((cb) => {
        try {
          cb({ type, data });
        } catch (e) {
          console.error(e);
        }
      });
    }
  }
}

export function useEventStream<T = any>(
  eventType: EventType | '*',
  callback: (data: T) => void
) {
  const savedCallback = useRef(callback);
  savedCallback.current = callback;

  useEffect(() => {
    const manager = EventStreamManager.getInstance();
    const unsubscribe = manager.subscribe<T>(eventType, (data) => {
      savedCallback.current(data);
    });

    return () => {
      unsubscribe();
    };
  }, [eventType]);
}
