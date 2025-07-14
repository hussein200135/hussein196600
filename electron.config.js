
const { app } = require('electron');
const path = require('path');

module.exports = {
  // Main process file
  main: path.join(__dirname, 'dist-electron/main-electron.js'),
  
  // Preload script
  preload: path.join(__dirname, 'dist-electron/preload.js'),
  
  // Build configuration
  build: {
    appId: 'com.smartmedicallab.desktop',
    productName: 'المختبر الذكي - Smart Medical Lab',
    
    directories: {
      output: 'release',
      buildResources: 'build'
    },
    
    files: [
      'dist/**/*',
      'dist-electron/**/*',
      'ai-backend/**/*',
      'node_modules/**/*',
      '!node_modules/.cache',
      '!node_modules/.vite'
    ],
    
    extraResources: [
      {
        from: 'ai-backend',
        to: 'ai-backend',
        filter: ['**/*', '!**/*.pyc', '!**/__pycache__']
      }
    ],
    
    // Windows configuration
    win: {
      target: [
        {
          target: 'nsis',
          arch: ['x64', 'ia32']
        },
        {
          target: 'portable',
          arch: ['x64']
        }
      ],
      icon: 'build/icon.ico',
      publisherName: 'Smart Medical Lab'
    },
    
    // macOS configuration
    mac: {
      target: [
        {
          target: 'dmg',
          arch: ['x64', 'arm64']
        }
      ],
      icon: 'build/icon.icns',
      category: 'public.app-category.medical',
      hardenedRuntime: true,
      gatekeeperAssess: false
    },
    
    // Linux configuration
    linux: {
      target: [
        {
          target: 'AppImage',
          arch: ['x64']
        },
        {
          target: 'deb',
          arch: ['x64']
        }
      ],
      icon: 'build/icon.png',
      category: 'Science',
      synopsis: 'نظام المختبر الذكي للتحليل الطبي'
    },
    
    // NSIS installer configuration (Windows)
    nsis: {
      oneClick: false,
      allowElevation: true,
      allowToChangeInstallationDirectory: true,
      createDesktopShortcut: true,
      createStartMenuShortcut: true,
      shortcutName: 'المختبر الذكي',
      include: 'build/installer.nsh'
    },
    
    // DMG configuration (macOS)
    dmg: {
      title: 'المختبر الذكي ${version}',
      background: 'build/dmg-background.png',
      iconSize: 100,
      contents: [
        {
          x: 380,
          y: 280,
          type: 'link',
          path: '/Applications'
        },
        {
          x: 110,
          y: 280,
          type: 'file'
        }
      ],
      window: {
        width: 540,
        height: 400
      }
    },
    
    // Auto updater
    publish: {
      provider: 'github',
      owner: 'smart-medical-lab',
      repo: 'desktop-app'
    }
  }
};
