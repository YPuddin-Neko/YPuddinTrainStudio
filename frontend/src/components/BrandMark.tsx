/** A small, code-native pudding mark that stays legible in the navigation rail. */
export default function BrandMark() {
  return <svg className="sidebar-brand-mark" width="30" height="30" viewBox="0 0 32 32" fill="none" aria-hidden="true">
    <rect x=".5" y=".5" width="31" height="31" rx="10" fill="#F7ECD9"/>
    <path d="M10.2 10.5C10.5 9 12.8 8 16 8s5.5 1 5.8 2.5l2.3 11.2c.4 2.1-3.3 3.8-8.1 3.8s-8.5-1.7-8.1-3.8l2.3-11.2Z" fill="#E3AB68"/>
    <path d="M10.2 10.5C10.5 12.2 21.5 12.2 21.8 10.5L23.7 20c.4 2-3.1 3.5-7.7 3.5s-8.1-1.5-7.7-3.5l1.9-9.5Z" fill="#FFE6AC"/>
    <ellipse cx="16" cy="10.4" rx="5.8" ry="2.6" fill="#A76A42"/>
    <path d="m12 13.8-.9 5.6" stroke="#FFF4D7" strokeWidth="1.8" strokeLinecap="round"/>
    <path d="M13.2 8.5c1.3-.4 3-.5 4.4-.2" stroke="#D69765" strokeWidth="1.3" strokeLinecap="round"/>
  </svg>;
}
