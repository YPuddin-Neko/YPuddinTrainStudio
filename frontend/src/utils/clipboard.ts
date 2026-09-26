/** Copy text, also on plain-HTTP LAN addresses where the Clipboard API is missing. */
export async function copyText(value: string): Promise<void> {
  if (navigator.clipboard && window.isSecureContext) { await navigator.clipboard.writeText(value); return; }
  const area = document.createElement('textarea');
  area.value = value; area.setAttribute('readonly', ''); area.style.position = 'fixed'; area.style.opacity = '0';
  document.body.appendChild(area); area.select();
  try { if (!document.execCommand('copy')) throw new Error('copy failed'); } finally { area.remove(); }
}
