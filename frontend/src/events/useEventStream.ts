import { useEffect, useRef, useState } from 'react';
import { apiUrl } from '../api/client';
import { EventType, EVENT_TYPES } from './eventTypes';

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected';

type EventCallback<T = any> = (data: T) => void;

class EventStreamManager {
  private static instance: EventStreamManager;
  private eventSource: EventSource | null = null;
  private listeners: Map<string, Set<EventCallback>> = new Map();
  private lastEventId: string = '';
  private reconnectTimer: any = null;
  private isConnecting: boolean = false;
  private activityTimer: ReturnType<typeof setInterval> | null = null;
  private lastActivityAt = 0;

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
    if (this.activityTimer) clearInterval(this.activityTimer);
    this.activityTimer = null;
    this.status = 'disconnected';
  }

  private reconnect(source: EventSource) {
    // Late callbacks from a discarded connection must not close its replacement.
    if (this.eventSource !== source) return;
    source.close();
    this.eventSource = null;
    this.isConnecting = false;
    if (this.activityTimer) clearInterval(this.activityTimer);
    this.activityTimer = null;
    this.setStatus('disconnected');
    if (!this.reconnectTimer) {
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        this.ensureConnection();
      }, 3000);
    }
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

    const url = new URL(apiUrl('/events'));
    if (this.lastEventId) {
      url.searchParams.set('last_event_id', this.lastEventId);
    }

    try {
      const source = new EventSource(url.toString());
      this.eventSource = source;
      this.lastActivityAt = Date.now();
      // Telemetry can be configured up to 60 seconds. Allow that interval before
      // replacing a proxy connection that stayed open after the server disappeared.
      this.activityTimer = setInterval(() => {
        if (Date.now() - this.lastActivityAt >= 75_000) this.reconnect(source);
      }, 5000);

      source.onopen = () => {
        if (this.eventSource !== source) return;
        this.lastActivityAt = Date.now();
        this.isConnecting = false;
        this.setStatus('connected');
        if (this.reconnectTimer) {
          clearTimeout(this.reconnectTimer);
          this.reconnectTimer = null;
        }
      };

      // 监听所有预定义的事件类型
      Object.values(EVENT_TYPES).forEach((eventType) => {
        source.addEventListener(eventType, (e: MessageEvent) => {
          if (this.eventSource !== source) return;
          this.lastActivityAt = Date.now();
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

      source.onerror = () => this.reconnect(source);
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
