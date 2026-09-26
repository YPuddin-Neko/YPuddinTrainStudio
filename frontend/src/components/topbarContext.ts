import React from 'react';

/** The top bar's location slot; null outside the app shell. */
export const TopbarContext = React.createContext<HTMLElement | null>(null);
