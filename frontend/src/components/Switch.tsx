import React from 'react';
import './switch.css';

type Props = Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type' | 'role' | 'checked' | 'onChange' | 'children'> & {
  checked: boolean;
  onCheckedChange?: (checked: boolean) => void;
  /** Visible text beside the switch; it names the switch unless aria-label is given. */
  children?: React.ReactNode;
};

/** On/off control. Multi-select lists keep ordinary checkboxes. */
export default function Switch({ checked, onCheckedChange, children, className = '', disabled, readOnly, ...input }: Props) {
  return <label className={`studio-switch ${className}`} data-state={checked ? 'on' : 'off'}>
    <input {...input} type="checkbox" role="switch" checked={checked} disabled={disabled} readOnly={readOnly}
      onChange={event => { if (!readOnly) onCheckedChange?.(event.target.checked); }}/>
    <span className="studio-switch-track" aria-hidden="true"><span className="studio-switch-thumb"/></span>
    {children != null && <span className="studio-switch-text">{children}</span>}
  </label>;
}
