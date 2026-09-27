import React from 'react';

/** Settings pages report the service's restart_required flag; the settings shell shows it beside the title. */
export const RestartRequiredContext = React.createContext<(required: boolean) => void>(() => {});
