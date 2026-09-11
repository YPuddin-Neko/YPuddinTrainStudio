import { useTranslation } from 'react-i18next';

/** Project workflow copy stays paired while the shared locale catalog evolves. */
export function useWorkspaceText() {
  const { i18n } = useTranslation();
  return (zh: string, en: string) => i18n.resolvedLanguage?.startsWith('en') ? en : zh;
}
