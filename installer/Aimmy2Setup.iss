; King Aim single-machine installer.
; Bundles the self-contained published app (no .NET runtime/SDK needed on the
; target machine) and chains the ViGEmBus driver installer, which the gamepad
; assist feature requires to create a virtual Xbox 360 controller.
;
; Build order:
;   1. dotnet publish ..\Aimmy2\Aimmy2.csproj -c Release -r win-x64 --self-contained true -o ..\publish\Aimmy2
;   2. Ensure redist\ViGEmBus_1.22.0_x64_x86_arm64.exe is present
;   3. Compile this script with ISCC (Inno Setup 6+)

#define MyAppName "King Aim"
#define MyAppVersion "2.5.0"
#define MyAppPublisher "King Aim"
#define MyAppExeName "YmmiaV2.exe"
#ifndef PublishDir
  #define PublishDir "..\publish\Aimmy2"
#endif
#ifndef StableBundleDir
  #define StableBundleDir "payload\stable-detector"
#endif
#ifndef InstallerOutputDir
  #define InstallerOutputDir "output"
#endif
#define RedistDir "redist"

[Setup]
AppId={{A1F3C2D4-8E5B-4A9C-B7D6-3F2E1A0C9B87}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir={#InstallerOutputDir}
OutputBaseFilename=KingAimSetup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
DisableDirPage=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
; Published app — self-contained win-x64: includes .NET runtime, no separate install needed.
; Includes all AI models under Models\ (ApexLegendsV1, BadBusiness30K, TheFinals20K,
; TheFinals12K, CounterStrike5K, Rust5K, UniversalV4, AIOHumanoidV7, CounterBloxHead,
; BattleBitV1, EnemyDetectionV1) and their manifest.json files.
Source: "{#PublishDir}\*"; DestDir: "{app}"; Excludes: "Models\*,bin\configs\*"; Flags: ignoreversion recursesubdirs createallsubdirs

; Verified stable bundle. Build-Installer.ps1 supplies this payload explicitly.
Source: "{#StableBundleDir}\model.onnx"; DestDir: "{app}\Models\stable-detector"; Flags: ignoreversion
Source: "{#StableBundleDir}\manifest.json"; DestDir: "{app}\Models\stable-detector"; Flags: ignoreversion
Source: "{#StableBundleDir}\checksums.ini"; DestDir: "{app}\Models\stable-detector"; Flags: ignoreversion
Source: "{#StableBundleDir}\checksums.json"; DestDir: "{app}\Models\stable-detector"; Flags: ignoreversion

; ViGEmBus driver installer — required for virtual gamepad output (right-stick aim assist).
; App degrades gracefully (IsConnected=false) if driver is not present.
#ifndef SkipRedist
  Source: "{#RedistDir}\ViGEmBus_1.22.0_x64_x86_arm64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
#endif

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Install/upgrade ViGEmBus silently. Required for gamepad assist output.
#ifndef SkipRedist
  Filename: "{tmp}\ViGEmBus_1.22.0_x64_x86_arm64.exe"; Parameters: "/quiet /norestart"; StatusMsg: "Installing virtual gamepad driver (ViGEmBus)..."; Flags: waituntilterminated
#endif

Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Leave user-added models and configs on uninstall — only remove generated data folders.
Type: filesandordirs; Name: "{app}\bin\images"
Type: filesandordirs; Name: "{app}\bin\labels"
