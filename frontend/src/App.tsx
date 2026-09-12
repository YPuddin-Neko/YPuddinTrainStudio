import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import AppRoutes from './router';
import RouteError from './components/RouteError';

const router = createBrowserRouter([{ path: '*', element: <AppRoutes />, errorElement: <RouteError /> }]);

export default function App() {
  return <RouterProvider router={router} />;
}
