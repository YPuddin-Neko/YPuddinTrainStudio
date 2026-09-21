import { render, screen } from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';
import { beforeEach, expect, it } from 'vitest';
import RouteError from '../../../frontend/src/components/RouteError';
import i18n from '../../../frontend/src/i18n';

beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });

it('provides a usable recovery screen when a previously open app requests an obsolete lazy bundle', async () => {
  const router = createMemoryRouter([{path:'/',loader:() => { throw new TypeError('Failed to fetch dynamically imported module: /assets/old.js'); }, errorElement:<RouteError/>}]);
  render(<RouterProvider router={router}/>);
  expect(await screen.findByRole('heading',{name:'页面资源未能加载'})).toBeInTheDocument();
  expect(screen.getByRole('button',{name:'重新载入'})).toBeInTheDocument();
  expect(screen.queryByText('Unexpected Application Error!')).not.toBeInTheDocument();
});
