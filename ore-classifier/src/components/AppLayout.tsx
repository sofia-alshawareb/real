import { useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Drawer from '@mui/material/Drawer';
import Box from '@mui/material/Box';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Tooltip from '@mui/material/Tooltip';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';
import IconButton from '@mui/material/IconButton';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import Divider from '@mui/material/Divider';
import ScienceIcon from '@mui/icons-material/Science';
import InventoryIcon from '@mui/icons-material/Inventory2';
import MapIcon from '@mui/icons-material/Map';
import DashboardIcon from '@mui/icons-material/Dashboard';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import { useSettingsStore } from '../stores/settingsStore';
import { resetDemoData } from '../services/seed/seedData';
import { ConfirmDialog } from './ConfirmDialog';
import { notify } from '../utils/toast';

const DRAWER_WIDTH = 232;

const NAV_ITEMS = [
  { to: '/', label: 'Эксперименты', icon: <ScienceIcon /> },
  { to: '/deposits', label: 'Справочник месторождений', icon: <MapIcon /> },
  { to: '/dashboard', label: 'Дашборд', icon: <DashboardIcon /> },
];

export function AppLayout() {
  const location = useLocation();
  const authorName = useSettingsStore((s) => s.authorName);
  const setAuthorName = useSettingsStore((s) => s.setAuthorName);
  const mlOffline = useSettingsStore((s) => s.mlOffline);
  const setMlOffline = useSettingsStore((s) => s.setMlOffline);
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [resetOpen, setResetOpen] = useState(false);
  const [resetting, setResetting] = useState(false);

  const handleReset = async () => {
    setResetting(true);
    try {
      await resetDemoData();
      notify('Демо-данные сброшены и пересозданы', 'success');
    } finally {
      setResetting(false);
      setResetOpen(false);
    }
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <AppBar position="fixed" color="inherit" sx={{ zIndex: (t) => t.zIndex.drawer + 1, backgroundColor: '#fff' }}>
        <Toolbar sx={{ gap: 2 }}>
          <InventoryIcon color="primary" />
          <Box sx={{ mr: 2, minWidth: 0 }}>
            <Typography
              variant="subtitle1"
              noWrap
              component="div"
              sx={{
                fontWeight: 700,
                color: 'primary.main',
                lineHeight: 1.2,
              }}
            >
              Команда Real — Норникель AI Science Hack
            </Typography>
            <Typography
              variant="caption"
              noWrap
              component="div"
              sx={{
                color: 'text.secondary',
                lineHeight: 1.2,
              }}
            >
              Классификация руды по OM-шлифам
            </Typography>
          </Box>
          <Box sx={{ flexGrow: 1 }} />
          <TextField
            label="Автор"
            size="small"
            value={authorName}
            onChange={(e) => setAuthorName(e.target.value)}
            placeholder="Введите имя"
            sx={{ width: 220 }}
          />
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <FormControlLabel
              control={<Switch checked={mlOffline} onChange={(e) => setMlOffline(e.target.checked)} color="warning" />}
              label="Демо: имитация сбоя анализа"
            />
            <Tooltip
              title="Включите, чтобы имитировать недоступность сервиса ML-анализа: новые кадры уходят в очередь и ждут либо ручной разметки, либо восстановления сервиса (переключатель снова в положение «выкл»)."
            >
              <InfoOutlinedIcon fontSize="small" sx={{ color: 'text.secondary', cursor: 'help' }} />
            </Tooltip>
          </Box>
          <IconButton onClick={(e) => setMenuAnchor(e.currentTarget)}>
            <MoreVertIcon />
          </IconButton>
          <Menu anchorEl={menuAnchor} open={Boolean(menuAnchor)} onClose={() => setMenuAnchor(null)}>
            <MenuItem
              onClick={() => {
                setMenuAnchor(null);
                setResetOpen(true);
              }}
            >
              <ListItemIcon>
                <RestartAltIcon fontSize="small" />
              </ListItemIcon>
              Сбросить демо-данные
            </MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>
      <Drawer
        variant="permanent"
        sx={{
          width: DRAWER_WIDTH,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: { width: DRAWER_WIDTH, boxSizing: 'border-box' },
        }}
      >
        <Toolbar />
        <List sx={{ px: 1, pt: 1 }}>
          {NAV_ITEMS.map((item) => (
            <ListItemButton
              key={item.to}
              component={NavLink}
              to={item.to}
              selected={location.pathname === item.to}
              sx={{ borderRadius: 1, mb: 0.5 }}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.label} />
            </ListItemButton>
          ))}
        </List>
        <Divider sx={{ mt: 'auto' }} />
        <Box sx={{
          p: 2
        }}>
          <Typography variant="caption" sx={{
            color: "text.secondary"
          }}>
            Прототип. Все данные хранятся локально в браузере.
          </Typography>
        </Box>
      </Drawer>
      <Box component="main" sx={{ flexGrow: 1, bgcolor: 'background.default', minHeight: '100vh' }}>
        <Toolbar />
        <Box sx={{
          p: 3
        }}>
          <Outlet />
        </Box>
      </Box>
      <ConfirmDialog
        open={resetOpen}
        title="Сбросить демо-данные?"
        description="Все эксперименты, месторождения и локальные изменения будут удалены и пересозданы заново. Действие необратимо."
        confirmLabel={resetting ? 'Сброс...' : 'Сбросить'}
        danger
        onConfirm={handleReset}
        onCancel={() => setResetOpen(false)}
      />
    </Box>
  );
}
