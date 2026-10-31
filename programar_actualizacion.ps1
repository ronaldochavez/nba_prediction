# Registra en el Programador de tareas de Windows la actualización automática de los datos (auto_update.py).
# La tarea corre cada 5 minutos mientras tu sesión de Windows está abierta y auto_update.py decide si toca
# descargar. Necesita el entorno .venv con las dependencias instaladas (ver README).
#   Instalar:  powershell -ExecutionPolicy Bypass -File programar_actualizacion.ps1
#   Quitar:    Unregister-ScheduledTask -TaskName "NBA - actualizar datos" -Confirm:$false
$root = $PSScriptRoot
$action = New-ScheduledTaskAction -Execute "$root\.venv\Scripts\pythonw.exe" -Argument "auto_update.py" -WorkingDirectory $root
# Desde la medianoche de hoy para que las corridas caigan en minutos redondos (:00, :05, :10...)
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 5)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
Register-ScheduledTask -TaskName "NBA - actualizar datos" -Action $action -Trigger $trigger -Settings $settings `
    -Description "Descarga los datos de la NBA y los sube a GitHub para la app de Streamlit Cloud (nba_prediction)" `
    -Force | Out-Null
Write-Host "Listo: la tarea 'NBA - actualizar datos' corre cada 5 minutos."
