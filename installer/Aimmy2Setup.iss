; Aimmy2 single-machine installer.
; Bundles the self-contained published app (no .NET runtime/SDK needed on the
; target machine) and chains the ViGEmBus driver installer, which the gamepad
; assist feature requires to create a virtual Xbox 360 controller.
;
; Build order:
;   1. dotnet publish ..\Aimmy2\Aimmy2.csproj -c Release -r win-x64 --self-contained true -o ..\publish\Aimmy2
;   2. Ensure ..\publish\Aimmy2\bin\models contains at least one .onnx + its manifest
;   3. Ensure redist\ViGEmBus_1.22.0_x64_x86_arm64.exe is present
;   4. Compile this script with ISCC (Inno Setup 6+)

#define MyAppName "Aimmy2"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Aimmy2"
#define MyAppExeName "YmmiaV2.exe"
#define PublishDir "..\publish\Aimmy2"
#define RedistDir "redist"

[Setup]
AppId={{B6C1B1B4-7C2E-4B7B-9A3A-9F5C8E2F1D01}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=output
OutputBaseFilename=Aimmy2Setup
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
; Published app (self-contained: includes the .NET runtime, no separate SDK/runtime install needed)
Source: "{#PublishDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; ViGEmBus driver installer, run silently after the app files are in place
Source: "{#RedistDir}\ViGEmBus_1.22.0_x64_x86_arm64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Install/upgrade the ViGEmBus driver silently. Required for gamepad assist output;
; the app itself still runs and degrades gracefully (IsConnected=false) if this is skipped.
Filename: "{tmp}\ViGEmBus_1.22.0_x64_x86_arm64.exe"; Parameters: "/quiet /norestart"; StatusMsg: "Installing virtual gamepad driver (ViGEmBus)..."; Flags: waituntilterminated

Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; bin\models / bin\configs etc. hold user-added models and configs — leave them on uninstall
; rather than silently deleting a user's downloaded models. Only remove the app binaries.
Type: filesandordirs; Name: "{app}\bin\images"
Type: filesandordirs; Name: "{app}\bin\labels"
