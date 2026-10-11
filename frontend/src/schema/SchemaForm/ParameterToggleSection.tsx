import type { ReactNode } from 'react';
import Switch from '../../components/Switch';

export default function ParameterToggleSection({ id, title, description, checked, onCheckedChange, children, help, disabled = false, parametersEnabled = checked, switchLabel, fieldPath, error, switchPosition = 'end', keepHelpEnabled = false }: {
  id: string;
  title: string;
  description?: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  children: ReactNode;
  help?: ReactNode;
  disabled?: boolean;
  parametersEnabled?: boolean;
  switchLabel?: string;
  fieldPath?: string;
  error?: string;
  switchPosition?: 'start' | 'end';
  keepHelpEnabled?: boolean;
}) {
  const toggle = <div data-testid={fieldPath ? `field-${fieldPath}` : undefined} data-field-path={fieldPath}>
    <Switch id={switchPosition === 'start' ? `${id}-switch` : undefined} aria-label={switchLabel || title} aria-describedby={description ? `${id}-description` : undefined} aria-controls={`${id}-settings`}
      checked={checked} disabled={disabled} onCheckedChange={onCheckedChange}/>
  </div>;
  return <section className="parameter-toggle-layout" data-enabled={checked} data-switch-position={switchPosition === 'start' ? 'start' : undefined} aria-labelledby={`${id}-title`}>
    <header className="parameter-toggle-heading">
      <div className="parameter-toggle-copy">
        <div className="parameter-toggle-title">
          {switchPosition === 'start' && toggle}
          <h3 id={`${id}-title`}>{switchPosition === 'start' ? <label htmlFor={`${id}-switch`}>{title}</label> : title}</h3>{help}
        </div>
        {description && <p id={`${id}-description`}>{description}</p>}
        {error && <p role="alert" className="config-field-error">{error}</p>}
      </div>
      {switchPosition === 'end' && toggle}
    </header>
    <fieldset id={`${id}-settings`} disabled={disabled || (!parametersEnabled && !keepHelpEnabled)}
      data-disabled={(keepHelpEnabled && !parametersEnabled) || undefined} className="parameter-toggle-body">
      {children}
    </fieldset>
  </section>;
}
