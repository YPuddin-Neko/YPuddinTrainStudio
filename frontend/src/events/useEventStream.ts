import { useEffect, useRef, useState } from 'react';
import { apiUrl } from '../api/client';
import { EventType, EVENT_TYPES } from './eventTypes';
import { createMockEventSource, shouldUseMockEvents } from './mockEventSource';

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected';

type EventCallback<T = any> = (data: T) => void;

class EventStreamManager {
  private static instance: EventStreamManager;
  private eventSource: EventSource | null = null;
  private listeners: Map<string, Set<EventCallback>> = new Map();
  private lastEventId: string = '';
  private reconnectTimer: any = null;
  private isConnecting: boolean = false;

  private status: ConnectionStatus = 'connecting';
  private statusListeners = new Set<(status: ConnectionStatus) => void>();

  public subscribeStatus(callback: (status: ConnectionStatus) => void) {
    this.statusListeners.add(callback);
    callback(this.status);
    this.ensureConnection();
    return () => { this.statusListeners.delete(callback); this.releaseIfUnused(); };
  }

  private setStatus(status: ConnectionStatus) {
    this.status = status;
    this.statusListeners.forEach((callback) => callback(status));
  }

  private releaseIfUnused() {
    if (this.listeners.size || this.statusListeners.size) return;
    this.eventSource?.close();
    this.eventSource = null;
    this.isConnecting = false;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
    this.status = 'disconnected';
  }

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
      this.releaseIfUnused();
    };
  }

  private ensureConnection() {
    if (this.eventSource || this.isConnecting || typeof window === 'undefined') return;

    this.isConnecting = true;
    this.setStatus('connecting');

    // 开发态 mock 模式：用本地事件生成器代替真实 EventSource（MSW 无法拦截 SSE）
    if (shouldUseMockEvents()) {
      const mock = createMockEventSource();
      this.eventSource = mock as any;
      Object.values(EVENT_TYPES).forEach((eventType) => {
        mock.addEventListener(eventType, (e: { data: string; lastEventId: string }) => {
          this.lastEventId = e.lastEventId;
          try {
            this.dispatch(eventType, JSON.parse(e.data));
          } catch (err) {
            console.error(`Failed to parse mock SSE event data for ${eventType}:`, err);
          }
        });
      });
      this.isConnecting = false;
      this.setStatus('connected');
      return;
    }

    const url = new URL(apiUrl('/events'));
    if (this.lastEventId) {
      url.searchParams.set('last_event_id', this.lastEventId);
    }

    try {
      this.eventSource = new EventSource(url.toString());

      this.eventSource.onopen = () => {
        this.isConnecting = false;
        this.setStatus('connected');
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
        this.setStatus('disconnected');
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
    } catch {
      this.isConnecting = false;
      this.setStatus('disconnected');
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

export function useEventStreamStatus() {
  const [status, setStatus] = useState<ConnectionStatus>('connecting');
  useEffect(() => EventStreamManager.getInstance().subscribeStatus(setStatus), []);
  return status;
}
