import React, { useEffect, useMemo, useRef, useState } from 'react';
import { CameraView, useCameraPermissions } from 'expo-camera';
import {
  Platform,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

const tabLabels = ['Dashboard', 'Tuner', 'Intonation', 'Coach', 'Settings'] as const;
type TabKey = (typeof tabLabels)[number];

type StringData = {
  name: string;
  note: string;
  target: number;
  current: number;
  tolerance: number;
};

const strings: StringData[] = [
  { name: 'G', note: 'G3', target: 196.0, current: 196.0, tolerance: 0.6 },
  { name: 'D', note: 'D4', target: 293.66, current: 293.66, tolerance: 1.0 },
  { name: 'A', note: 'A4', target: 440.0, current: 440.0, tolerance: 1.2 },
  { name: 'E', note: 'E5', target: 659.25, current: 659.25, tolerance: 1.8 },
];

const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

function frequencyToNoteName(frequency: number): string {
  const midi = Math.round(69 + 12 * Math.log2(frequency / 440));
  const noteIndex = ((midi % 12) + 12) % 12;
  return noteNames[noteIndex];
}

function frequencyToCents(frequency: number, target: number): number {
  return Math.round((1200 * Math.log2(frequency / target)) / 1) * 1;
}

function estimatePitch(buffer: Float32Array, sampleRate: number): number {
  if (buffer.length < 32) return 0;

  let bestOffset = -1;
  let bestCorrelation = 0;
  const minFrequency = 50;
  const maxFrequency = 1500;
  const minSamples = Math.floor(sampleRate / maxFrequency);
  const maxSamples = Math.floor(sampleRate / minFrequency);

  for (let offset = minSamples; offset <= maxSamples; offset += 1) {
    let correlation = 0;
    for (let i = 0; i < buffer.length - offset; i += 1) {
      correlation += buffer[i] * buffer[i + offset];
    }

    if (correlation > bestCorrelation) {
      bestCorrelation = correlation;
      bestOffset = offset;
    }
  }

  if (bestOffset <= 0) {
    return 0;
  }

  return sampleRate / bestOffset;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('Dashboard');
  const [selectedString, setSelectedString] = useState<string>('A');
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [micPermission, setMicPermission] = useState<boolean>(false);
  const [micEnabled, setMicEnabled] = useState<boolean>(false);
  const [detectedPitch, setDetectedPitch] = useState<number>(440);
  const [detectedNote, setDetectedNote] = useState<string>('A');
  const [detectedCents, setDetectedCents] = useState<number>(0);
  const [poseStatus, setPoseStatus] = useState<string>('Violine erkannt');
  const [webCameraStream, setWebCameraStream] = useState<MediaStream | null>(null);
  const cameraVideoRef = useRef<any>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const selectedStringData = useMemo(
    () => strings.find((item) => item.name === selectedString) ?? strings[2],
    [selectedString],
  );

  useEffect(() => {
    if (Platform.OS !== 'web' && cameraPermission && !cameraPermission.granted && cameraPermission.canAskAgain) {
      requestCameraPermission();
    }
  }, [cameraPermission, requestCameraPermission]);

  useEffect(() => {
    if (Platform.OS === 'web' && webCameraStream && cameraVideoRef.current) {
      cameraVideoRef.current.srcObject = webCameraStream;
      cameraVideoRef.current.play();
    }
  }, [webCameraStream]);

  useEffect(() => {
    if (Platform.OS !== 'web') {
      return;
    }

    const startWebCamera = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' },
        audio: true,
      });

      setWebCameraStream(stream);

      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.2;
      source.connect(analyser);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;
      setMicPermission(true);
      setMicEnabled(true);

      const tick = () => {
        const currentAnalyser = analyserRef.current;
        if (!currentAnalyser) {
          return;
        }

        const buffer = new Float32Array(currentAnalyser.fftSize);
        currentAnalyser.getFloatTimeDomainData(buffer);
        const pitch = estimatePitch(buffer, audioContext.sampleRate);

        if (pitch > 60 && pitch < 1200) {
          setDetectedPitch(pitch);
          setDetectedNote(frequencyToNoteName(pitch));
          setDetectedCents(frequencyToCents(pitch, selectedStringData.target));
        }

        animationFrameRef.current = requestAnimationFrame(tick);
      };

      tick();
    };

    startWebCamera();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      if (webCameraStream) {
        webCameraStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [selectedStringData.target]);

  useEffect(() => {
    if (Platform.OS === 'web') {
      const status = Math.abs(detectedCents) < 8 ? 'Violine erkannt · korrekt' : Math.abs(detectedCents) < 25 ? 'Violine erkannt · leicht daneben' : 'Violine erkannt · deutlich daneben';
      setPoseStatus(status);
    }
  }, [detectedCents]);

  const handleCameraPermissionRequest = async () => {
    if (Platform.OS === 'web') {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          return;
        }
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user' },
          audio: true,
        });
        setWebCameraStream(stream);
        setMicPermission(true);
        setMicEnabled(true);
      } catch (error) {
        console.warn('Kamera-/Mikrofonzugriff fehlgeschlagen:', error);
      }
      return;
    }

    const permission = await requestCameraPermission();
    if (permission.granted) {
      setMicPermission(true);
      setMicEnabled(true);
    }
  };

  const liveFrequency = detectedPitch > 0 ? detectedPitch.toFixed(1) : selectedStringData.target.toFixed(1);
  const liveTarget = selectedStringData.target;
  const liveDelta = detectedPitch > 0 ? detectedCents : 0;
  const tuningState = Math.abs(liveDelta) <= 8 ? 'perfekt' : Math.abs(liveDelta) <= 25 ? 'leicht daneben' : 'deutlich daneben';

  const renderDashboard = () => (
    <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <View style={styles.eyebrowRow}>
        <Text style={styles.eyebrow}>Violine Coach</Text>
        <Text style={styles.livePill}>{micEnabled ? 'Live' : 'Bereit'}</Text>
      </View>

      <Text style={styles.heroTitle}>Kamera & Mikrofon</Text>
      <Text style={styles.subtleText}>Die Haltung wird im Bild erkannt und der Ton wird über das Mikrofon gemessen.</Text>

      <View style={styles.cameraCard}>
        <View style={styles.panelHeader}>
          <Text style={styles.panelTitle}>Live-Kameraanalyse</Text>
          {Platform.OS === 'web' || cameraPermission?.granted ? (
            <Text style={[styles.badge, { backgroundColor: '#dffaf0', color: '#1a9b6d' }]}>Aktiv</Text>
          ) : (
            <TouchableOpacity onPress={handleCameraPermissionRequest} style={styles.permissionButton}>
              <Text style={styles.permissionButtonText}>Kamera erlauben</Text>
            </TouchableOpacity>
          )}
        </View>

        {Platform.OS === 'web' ? (
          webCameraStream ? (
            <View style={styles.cameraPreviewContainer}>
              <video ref={cameraVideoRef} autoPlay muted playsInline style={styles.cameraVideo} />
              <View style={styles.poseBox}>
                <View style={styles.poseLineOne} />
                <View style={styles.poseLineTwo} />
                <View style={styles.poseLineThree} />
                <View style={styles.poseDot} />
              </View>
            </View>
          ) : (
            <TouchableOpacity onPress={handleCameraPermissionRequest} style={styles.permissionPanel}>
              <Text style={styles.permissionPanelText}>Kamera und Mikrofon aktivieren</Text>
            </TouchableOpacity>
          )
        ) : cameraPermission?.granted ? (
          <View style={styles.cameraPreviewContainer}>
            <CameraView facing="front" style={styles.cameraVideo} />
            <View style={styles.poseBox}>
              <View style={styles.poseLineOne} />
              <View style={styles.poseLineTwo} />
              <View style={styles.poseLineThree} />
              <View style={styles.poseDot} />
            </View>
          </View>
        ) : (
          <TouchableOpacity onPress={handleCameraPermissionRequest} style={styles.permissionPanel}>
            <Text style={styles.permissionPanelText}>Bitte Kamera erlauben</Text>
          </TouchableOpacity>
        )}
      </View>

      <View style={styles.metricGrid}>
        <View style={styles.metricCard}>
          <Text style={styles.metricTone}>{poseStatus}</Text>
          <Text style={styles.metricLabel}>Haltung</Text>
        </View>
        <View style={styles.metricCard}>
          <Text style={styles.metricTone}>{micEnabled ? detectedNote : selectedStringData.note}</Text>
          <Text style={styles.metricLabel}>Ton</Text>
        </View>
      </View>
    </ScrollView>
  );

  const renderTuner = () => (
    <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <Text style={styles.sectionTitle}>Stimmgerät</Text>

      <View style={styles.tunerCard}>
        <View style={styles.tunerHeader}>
          <Text style={styles.tunerLabel}>Saite</Text>
          <Text style={styles.tunerLabel}>Live</Text>
        </View>

        <View style={styles.tunerStrip}>
          {strings.map((item) => (
            <TouchableOpacity
              key={item.name}
              activeOpacity={0.8}
              onPress={() => setSelectedString(item.name)}
              style={[styles.stringButton, selectedString === item.name && styles.stringButtonActive]}
            >
              <Text style={selectedString === item.name ? styles.stringButtonTextActive : styles.stringButtonText}>{item.name}</Text>
            </TouchableOpacity>
          ))}
        </View>

        <View style={styles.tunerMain}>
          <Text style={styles.bigNote}>{selectedStringData.note}</Text>
          <Text style={styles.bigFreq}>{liveFrequency} Hz</Text>
          <Text style={styles.targetText}>Soll: {liveTarget.toFixed(2)} Hz</Text>
        </View>

        <View style={styles.gaugeBox}>
          <View style={styles.gaugeTrack}>
            <View
              style={[
                styles.gaugeIndicator,
                {
                  left: `${Math.min(Math.max(50 + liveDelta * 1.8, 8), 92)}%`,
                  backgroundColor: liveDelta === 0 ? '#38d39f' : Math.abs(liveDelta) <= 15 ? '#ffbf4d' : '#ff6b6b',
                },
              ]}
            />
          </View>
          <View style={styles.gaugeLabels}>
            <Text style={styles.gaugeLabel}>-30</Text>
            <Text style={styles.gaugeLabel}>0</Text>
            <Text style={styles.gaugeLabel}>+30</Text>
          </View>
        </View>

        <View style={styles.tunerFooter}>
          <View style={styles.feedbackPillRow}>
            <Text
              style={[
                styles.feedbackPill,
                {
                  backgroundColor: Math.abs(liveDelta) <= 8 ? '#dffaf0' : Math.abs(liveDelta) <= 25 ? '#fff2d9' : '#ffe5e5',
                  color: Math.abs(liveDelta) <= 8 ? '#1a9b6d' : Math.abs(liveDelta) <= 25 ? '#996300' : '#b3261e',
                },
              ]}
            >
              {tuningState}
            </Text>
            <Text style={styles.deltaValue}>{Math.abs(liveDelta)} Cent</Text>
          </View>
          <Text style={styles.directionText}>
            {liveDelta > 0 ? 'zu hoch' : liveDelta < 0 ? 'zu tief' : 'perfekt'}
          </Text>
        </View>
      </View>
    </ScrollView>
  );

  const renderIntonation = () => (
    <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <Text style={styles.sectionTitle}>Intonations-Feedback</Text>

      <View style={styles.panel}>
        <View style={styles.panelHeader}>
          <Text style={styles.panelTitle}>Aktueller Ton</Text>
          <Text style={styles.linkText}>{detectedNote || selectedStringData.note}</Text>
        </View>

        <View style={styles.intonationMain}>
          <View style={styles.noteBadge}>
            <Text style={styles.noteBadgeText}>{detectedNote || selectedStringData.note}</Text>
          </View>
          <View>
            <Text style={styles.bigFreq}>{liveFrequency} Hz</Text>
            <Text style={styles.statusText}>Abweichung: {liveDelta} Cent</Text>
          </View>
        </View>

        <View style={styles.tonesRow}>
          {['Perfekt', 'Leicht daneben', 'Deutlich daneben'].map((label, index) => (
            <View
              key={label}
              style={[
                styles.toneStatus,
                index === 0 && styles.toneStatusActive,
                index === 0 && { backgroundColor: '#dffaf0' },
              ]}
            >
              <Text style={[styles.toneStatusText, index === 0 && { color: '#12785c' }]}>{label}</Text>
            </View>
          ))}
        </View>
      </View>
    </ScrollView>
  );

  const renderCoach = () => (
    <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <Text style={styles.sectionTitle}>KI-Coach</Text>

      {[
        {
          title: 'Handgelenk',
          detail: 'Die Hand steht gerade. Das linke Handgelenk bleibt stabil und die Geige sitzt richtig im Kinnwinkel.',
          severity: 'info',
        },
        {
          title: 'Intonation',
          detail: 'Die aktuelle Tonhöhe wird über das Mikrofon gemessen. Das Ziel ist die ausgewählte Saite mit minimaler Abweichung zu treffen.',
          severity: 'warn',
        },
        {
          title: 'Stimmung',
          detail: 'Wenn du die Saite spielst, erkennt die App sofort die Frequenz und vergleicht sie mit der Sollfrequenz.',
          severity: 'alert',
        },
      ].map((item) => (
        <View key={item.title} style={styles.coachCard}>
          <View style={styles.coachHead}>
            <Text style={styles.coachTitle}>{item.title}</Text>
            <Text
              style={[
                styles.severityBadge,
                item.severity === 'alert' && styles.severityAlert,
                item.severity === 'warn' && styles.severityWarn,
              ]}
            >
              {item.severity === 'alert' ? 'Rot' : item.severity === 'warn' ? 'Gelb' : 'Info'}
            </Text>
          </View>
          <Text style={styles.coachText}>{item.detail}</Text>
        </View>
      ))}
    </ScrollView>
  );

  const renderSettings = () => (
    <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <Text style={styles.sectionTitle}>Einstellungen</Text>

      <View style={styles.panel}>
        {[
          ['Kameraanalyse', cameraPermission?.granted || Platform.OS === 'web' ? 'Aktiv' : 'Aus'],
          ['Audio-Analyse', micEnabled ? 'Aktiv' : 'Aus'],
          ['Vibration', 'An'],
          ['Datenschutz', 'Lokal'],
        ].map(([label, value]) => (
          <View key={label} style={styles.settingRow}>
            <Text style={styles.settingLabel}>{label}</Text>
            <Text style={styles.settingValue}>{value}</Text>
          </View>
        ))}
      </View>
    </ScrollView>
  );

  const renderScreen = () => {
    switch (activeTab) {
      case 'Dashboard':
        return renderDashboard();
      case 'Tuner':
        return renderTuner();
      case 'Intonation':
        return renderIntonation();
      case 'Coach':
        return renderCoach();
      case 'Settings':
        return renderSettings();
      default:
        return renderDashboard();
    }
  };

  return (
    <View style={styles.safeArea}>
      <StatusBar barStyle="dark-content" />
      <View style={styles.appShell}>
        {renderScreen()}

        <View style={styles.tabBar}>
          {tabLabels.map((tab) => (
            <TouchableOpacity
              key={tab}
              accessibilityRole="button"
              activeOpacity={0.8}
              onPress={() => setActiveTab(tab)}
              style={[styles.tab, activeTab === tab && styles.tabActive]}
            >
              <Text style={[styles.tabText, activeTab === tab && styles.tabTextActive]}>{tab}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#f3f4ff',
  },
  appShell: {
    flex: 1,
    paddingTop: 28,
    backgroundColor: '#f3f4ff',
  },
  content: {
    paddingHorizontal: 16,
    paddingBottom: 120,
  },
  eyebrowRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  eyebrow: {
    fontSize: 12,
    color: '#5864ff',
    fontWeight: '700',
    letterSpacing: 1.2,
    textTransform: 'uppercase',
  },
  livePill: {
    backgroundColor: '#e9f8f1',
    color: '#107d5b',
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 4,
    fontSize: 11,
    fontWeight: '700',
  },
  heroTitle: {
    fontSize: 30,
    fontWeight: '800',
    color: '#111827',
    marginBottom: 4,
  },
  subtleText: {
    fontSize: 14,
    color: '#667085',
    marginBottom: 18,
  },
  cameraCard: {
    backgroundColor: '#ffffff',
    borderRadius: 18,
    padding: 12,
    marginBottom: 18,
    shadowColor: '#d8dce6',
    shadowOpacity: 0.16,
    shadowRadius: 14,
    shadowOffset: { width: 0, height: 8 },
    elevation: 2,
  },
  cameraPreviewContainer: {
    position: 'relative',
    width: '100%',
    height: 320,
    maxHeight: 420,
    borderRadius: 18,
    overflow: 'hidden',
    backgroundColor: '#0f172a',
  },
  cameraVideo: {
    width: '100%',
    height: '100%',
    backgroundColor: '#0f172a',
  },
  poseBox: {
    position: 'absolute',
    left: '10%',
    right: '10%',
    top: '10%',
    bottom: '10%',
    borderWidth: 2,
    borderColor: '#5d7cff',
    borderRadius: 18,
  },
  poseLineOne: {
    position: 'absolute',
    width: '100%',
    height: 2,
    backgroundColor: '#5d7cff',
    top: '50%',
  },
  poseLineTwo: {
    position: 'absolute',
    width: 2,
    height: '100%',
    backgroundColor: '#5d7cff',
    left: '50%',
  },
  poseLineThree: {
    position: 'absolute',
    width: 70,
    height: 70,
    borderRadius: 35,
    borderWidth: 2,
    borderColor: '#5d7cff',
    left: '50%',
    top: '50%',
    marginLeft: -35,
    marginTop: -35,
  },
  poseDot: {
    position: 'absolute',
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#38d39f',
    left: '52%',
    top: '52%',
  },
  cameraPlaceholder: {
    color: '#465264',
    fontSize: 14,
    lineHeight: 20,
    paddingVertical: 12,
  },
  permissionButton: {
    backgroundColor: '#edf2ff',
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  permissionButtonText: {
    color: '#2c45dc',
    fontSize: 11,
    fontWeight: '700',
  },
  permissionPanel: {
    backgroundColor: '#eef2ff',
    borderRadius: 18,
    paddingVertical: 28,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 140,
  },
  permissionPanelText: {
    color: '#2c45dc',
    fontSize: 16,
    fontWeight: '700',
  },
  metricGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 18,
    justifyContent: 'space-between',
  },
  metricCard: {
    width: '48%',
    minWidth: 0,
    backgroundColor: '#ffffff',
    borderRadius: 18,
    paddingVertical: 18,
    paddingHorizontal: 14,
    marginBottom: 12,
    shadowColor: '#b9c0ff',
    shadowOpacity: 0.12,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 6 },
    elevation: 2,
  },
  metricTone: {
    fontSize: 16,
    lineHeight: 21,
    fontWeight: '800',
    marginBottom: 4,
    color: '#111827',
  },
  metricLabel: {
    fontSize: 12,
    color: '#586274',
    fontWeight: '600',
  },
  panel: {
    backgroundColor: '#ffffff',
    borderRadius: 22,
    padding: 16,
    marginBottom: 18,
    shadowColor: '#d8dce6',
    shadowOpacity: 0.18,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 8 },
    elevation: 2,
  },
  panelHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  panelTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#111827',
  },
  badge: {
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 6,
    fontSize: 11,
    fontWeight: '800',
    overflow: 'hidden',
  },
  linkText: {
    color: '#5d7cff',
    fontWeight: '700',
  },
  statusText: {
    fontSize: 14,
    lineHeight: 20,
    color: '#465264',
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 28,
    fontWeight: '800',
    color: '#111827',
    marginBottom: 18,
  },
  tunerCard: {
    backgroundColor: '#ffffff',
    borderRadius: 24,
    padding: 18,
    shadowColor: '#dfe5ff',
    shadowOpacity: 0.18,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 10 },
    elevation: 2,
  },
  tunerHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  tunerLabel: {
    fontSize: 12,
    color: '#6b7280',
    fontWeight: '700',
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  tunerStrip: {
    flexDirection: 'row',
    marginBottom: 22,
    gap: 8,
  },
  stringButton: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 12,
    backgroundColor: '#eef2ff',
    alignItems: 'center',
  },
  stringButtonActive: {
    backgroundColor: '#5864ff',
  },
  stringButtonText: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1c2333',
  },
  stringButtonTextActive: {
    color: '#ffffff',
    fontWeight: '800',
  },
  tunerMain: {
    alignItems: 'center',
    marginBottom: 18,
  },
  bigNote: {
    fontSize: 32,
    fontWeight: '800',
    color: '#111827',
  },
  bigFreq: {
    marginTop: 8,
    fontSize: 28,
    fontWeight: '800',
    color: '#111827',
  },
  targetText: {
    marginTop: 8,
    fontSize: 14,
    color: '#5b657d',
  },
  gaugeBox: {
    marginBottom: 16,
  },
  gaugeTrack: {
    height: 12,
    borderRadius: 999,
    backgroundColor: '#eef1f6',
    position: 'relative',
  },
  gaugeIndicator: {
    position: 'absolute',
    top: -4,
    width: 18,
    height: 18,
    borderRadius: 999,
    transform: [{ translateX: -9 }],
    borderWidth: 3,
    borderColor: '#ffffff',
  },
  gaugeLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  gaugeLabel: {
    fontSize: 12,
    color: '#687385',
  },
  tunerFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  feedbackPillRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  feedbackPill: {
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 6,
    fontSize: 11,
    fontWeight: '700',
    overflow: 'hidden',
  },
  deltaValue: {
    fontSize: 16,
    fontWeight: '800',
    color: '#111827',
  },
  directionText: {
    fontSize: 14,
    color: '#465264',
    fontWeight: '600',
  },
  intonationMain: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 18,
    marginBottom: 18,
  },
  noteBadge: {
    width: 78,
    height: 78,
    borderRadius: 22,
    backgroundColor: '#edf2ff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  noteBadgeText: {
    fontSize: 28,
    fontWeight: '800',
    color: '#2b3bb8',
  },
  tonesRow: {
    flexDirection: 'row',
    gap: 8,
  },
  toneStatus: {
    flex: 1,
    borderRadius: 12,
    paddingVertical: 10,
    alignItems: 'center',
    backgroundColor: '#f3f4f6',
  },
  toneStatusActive: {
    borderWidth: 1,
    borderColor: '#9fe7ca',
  },
  toneStatusText: {
    color: '#485364',
    fontWeight: '700',
    fontSize: 12,
  },
  coachCard: {
    backgroundColor: '#ffffff',
    borderRadius: 20,
    padding: 16,
    marginBottom: 14,
    shadowColor: '#dfe5ff',
    shadowOpacity: 0.14,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 8 },
    elevation: 2,
  },
  coachHead: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  coachTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: '#111827',
  },
  severityBadge: {
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 6,
    fontSize: 11,
    fontWeight: '700',
    overflow: 'hidden',
    color: '#0f172a',
    backgroundColor: '#e5e7eb',
  },
  severityAlert: {
    backgroundColor: '#ffe5e5',
    color: '#b3261e',
  },
  severityWarn: {
    backgroundColor: '#fff2d9',
    color: '#996300',
  },
  coachText: {
    color: '#465264',
    fontSize: 14,
    lineHeight: 20,
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#eef1f5',
  },
  settingLabel: {
    fontSize: 15,
    color: '#1f2937',
    fontWeight: '600',
  },
  settingValue: {
    fontSize: 13,
    color: '#5864ff',
    fontWeight: '700',
  },
  tabBar: {
    position: 'absolute',
    left: 12,
    right: 12,
    bottom: 16,
    backgroundColor: '#ffffff',
    borderRadius: 22,
    flexDirection: 'row',
    padding: 6,
    shadowColor: '#dfe5ff',
    shadowOpacity: 0.2,
    shadowRadius: 14,
    shadowOffset: { width: 0, height: 8 },
    elevation: 2,
  },
  tab: {
    flex: 1,
    paddingVertical: 9,
    paddingHorizontal: 1,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tabActive: {
    backgroundColor: '#edf2ff',
  },
  tabText: {
    color: '#5c697d',
    fontSize: 9,
    fontWeight: '700',
  },
  tabTextActive: {
    color: '#2e45e6',
  },
});
