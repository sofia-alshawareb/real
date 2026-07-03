import { createBrowserRouter } from 'react-router-dom';
import { AppLayout } from './components/AppLayout';
import { ExperimentsListPage } from './features/experiments/ExperimentsListPage';
import { CreateExperimentPage } from './features/createExperiment/CreateExperimentPage';
import { ExperimentCardPage } from './features/experimentCard/ExperimentCardPage';
import { FrameEditorPage } from './features/frameEditor/FrameEditorPage';
import { DepositsPage } from './features/deposits/DepositsPage';
import { ReportDraftPage } from './features/report/ReportDraftPage';
import { DashboardPage } from './features/dashboard/DashboardPage';

export const router = createBrowserRouter([
  {
    element: <AppLayout />,
    children: [
      { path: '/', element: <ExperimentsListPage /> },
      { path: '/experiments/new', element: <CreateExperimentPage /> },
      { path: '/experiments/:experimentId', element: <ExperimentCardPage /> },
      { path: '/experiments/:experimentId/frames/:frameId', element: <FrameEditorPage /> },
      { path: '/experiments/:experimentId/report', element: <ReportDraftPage /> },
      { path: '/deposits', element: <DepositsPage /> },
      { path: '/dashboard', element: <DashboardPage /> },
    ],
  },
]);
