import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

import zhCN from './locales/zh-CN.json';
import en from './locales/en.json';

const resources = {
  'zh-CN': { translation: zhCN },
  en: { translation: en },
};

const savedLng = typeof localStorage !== 'undefined' ? localStorage.getItem('i18nextLng') : null;

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: 'zh-CN',
    lng: savedLng || 'zh-CN',
    interpolation: {
      escapeValue: false,
      // 本项目 locales 全部使用单花括号插值（{name}），此处全局对齐
      prefix: '{',
      suffix: '}',
    },
  });

export default i18n;
