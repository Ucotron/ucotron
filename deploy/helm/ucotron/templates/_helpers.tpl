{{/*
Expand the name of the chart.
*/}}
{{- define "ucotron.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "ucotron.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "ucotron.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "ucotron.labels" -}}
helm.sh/chart: {{ include "ucotron.chart" . }}
{{ include "ucotron.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "ucotron.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ucotron.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "ucotron.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "ucotron.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
ConfigMap name
*/}}
{{- define "ucotron.configMapName" -}}
{{- include "ucotron.fullname" . }}-config
{{- end }}

{{/*
PVC name
*/}}
{{- define "ucotron.pvcName" -}}
{{- include "ucotron.fullname" . }}-data
{{- end }}

{{/*
Writer name (multi-instance)
*/}}
{{- define "ucotron.writerName" -}}
{{- include "ucotron.fullname" . }}-writer
{{- end }}

{{/*
Reader name (multi-instance)
*/}}
{{- define "ucotron.readerName" -}}
{{- include "ucotron.fullname" . }}-reader
{{- end }}

{{/*
Writer selector labels
*/}}
{{- define "ucotron.writerSelectorLabels" -}}
{{ include "ucotron.selectorLabels" . }}
app.kubernetes.io/component: writer
{{- end }}

{{/*
Reader selector labels
*/}}
{{- define "ucotron.readerSelectorLabels" -}}
{{ include "ucotron.selectorLabels" . }}
app.kubernetes.io/component: reader
{{- end }}

{{/*
Writer labels
*/}}
{{- define "ucotron.writerLabels" -}}
{{ include "ucotron.labels" . }}
app.kubernetes.io/component: writer
{{- end }}

{{/*
Reader labels
*/}}
{{- define "ucotron.readerLabels" -}}
{{ include "ucotron.labels" . }}
app.kubernetes.io/component: reader
{{- end }}

{{/*
Writer ConfigMap name
*/}}
{{- define "ucotron.writerConfigMapName" -}}
{{- include "ucotron.fullname" . }}-writer-config
{{- end }}

{{/*
Reader ConfigMap name
*/}}
{{- define "ucotron.readerConfigMapName" -}}
{{- include "ucotron.fullname" . }}-reader-config
{{- end }}
