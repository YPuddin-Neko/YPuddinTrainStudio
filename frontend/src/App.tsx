import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import AppRoutes from './router';

const router = createBrowserRouter([{ path: '*', element: <AppRoutes /> }]);

export default function App() {
  return <RouterProvider router={router} />;
}
