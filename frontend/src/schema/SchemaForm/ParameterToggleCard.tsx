import type { ReactNode } from 'react';
import Switch from '../../components/Switch';

export default function ParameterToggleCard({ id, title, description, checked, onCheckedChange, children, help, disabled = false, switchLabel, fieldPath, group, error }: {
  id: string;
  title: string;
  description: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  children: ReactNode;
  help?: ReactNode;
  disabled?: boolean;
  switchLabel?: string;
  fieldPath?: string;
  group?: string;
  error?: string;
}) {
  return <section className="parameter-toggle-card" data-enabled={checked} data-group={group} aria-labelledby={`${id}-title`}>
    <header className="parameter-toggle-heading">
      <div className="parameter-toggle-copy">
        <div className="parameter-toggle-title"><h3 id={`${id}-title`}>{title}</h3>{help}</div>
        <p id={`${id}-description`}>{description}</p>
        {error && <p role="alert" className="config-field-error">{error}</p>}
      </div>
      <div data-testid={fieldPath ? `field-${fieldPath}` : undefined} data-field-path={fieldPath}>
        <Switch aria-label={switchLabel || title} aria-describedby={`${id}-description`} aria-controls={`${id}-settings`}
          checked={checked} disabled={disabled} onCheckedChange={onCheckedChange}/>
      </div>
    </header>
    <fieldset id={`${id}-settings`} disabled={disabled || !checked} className="parameter-toggle-body">
      {children}
    </fieldset>
  </section>;
}
