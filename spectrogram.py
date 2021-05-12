from obspy import read, Stream

PATH = 'More_examples_SAC/'
LABEL = 'Explosions/'
DATE = '20201223104202/'
#SHOW = False
channels = ['E', 'N', 'Z']
files = {c :PATH + LABEL + DATE +'PPPP_HH'+c+'_CN.sac' for c in channels} # channels E, N, Z

traces = [read(files[c])[0] for c in channels]
st = Stream(traces)


st.plot(color='gray', tick_format='%I:%M %p',starttime=st[0].stats.starttime, endtime=st[0].stats.endtime)

# https://www.earthinversion.com/utilities/concatenating-daily-seismic-traces-and-plot-spectrogram/
st.spectrogram(log=False, samp_rate=200.00)

