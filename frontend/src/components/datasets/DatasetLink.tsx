import { Link, useLocation, type LinkProps } from 'react-router-dom';

/** Keep the originating version, tab and filters when opening a dataset. */
export default function DatasetLink({ state, ...props }: LinkProps) {
  const location = useLocation();
  return <Link {...props} state={{ ...state, datasetReturnTo: `${location.pathname}${location.search}${location.hash}` }}/>;
}
