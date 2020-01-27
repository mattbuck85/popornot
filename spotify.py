import pandas as pd
import numpy as np
import spotipy
import sys
import spotipy
import spotipy.util as util
from spotipy import oauth2
from collections import defaultdict

USERNAME = "Placeholder"

def auth():
    with open('spotify_credentials.txt') as fh:
        client_id, client_secret = fh.read().split()
    scopes = "playlist-read-collaborative"
    redirect_uri = "http://localhost/redirect.html"
    token = util.prompt_for_user_token(
        USERNAME, scopes, client_id=client_id,
        client_secret=client_secret, redirect_uri=redirect_uri
    )
    return spotipy.Spotify(auth=token)

sp = auth()

track_columns = [
    'artist', 'name', 'num_samples', 'duration', 
    'tempo', 'tempo_confidence', 'time_signature',
    'time_signature_confidence', 'key', 'key_confidence', 
    'mode', 'mode_confidence'
]
def get_track_data(playlist_id):
    track_data = []
    results = sp.user_playlist(USERNAME, playlist_id, fields="tracks,next")
    tracks = results['tracks']
    sections_list, segments_list = [], []

    for track in tracks['items']:
        analysis = sp.audio_analysis(track['track']['id'])
        sections = pd.DataFrame(analysis['sections'])
        segments = pd.DataFrame(analysis['segments'])
        analysis = analysis['track']
        track = track['track']
        artists = ','.join(artist['name'] for artist in track['artists'])
        track_name = track['name']
        sections['artists'], segments['artists'] = artists, artists
        sections['name'], segments['name'] = track_name, track_name
        track_data.append(
            [
                artists, track_name, analysis['num_samples'], analysis['duration'],
                analysis['tempo'], analysis['tempo_confidence'], analysis['time_signature'],
                analysis['time_signature_confidence'], analysis['key'], analysis['key_confidence'],
                analysis['mode'], analysis['mode_confidence']
            ]
        )
        sections_list.append(sections)
        segments_list.append(segments)

    return tracks['items'], pd.DataFrame(track_data, columns=track_columns), pd.concat(sections_list), pd.concat(segments_list)

def get_track_features(tracks):
    feature_gen = (
        tracks[i:i+50] for i in range(0, len(tracks), 50)
    )
    analyzed = []
    
    for tracks_subset in feature_gen:
        analyzed += sp.audio_features(tracks=[track['track']['id'] for track in tracks_subset])
    while analyzed[-1] is None:
        analyzed.pop()

    df = pd.DataFrame(analyzed)
    df = df[
        list(set(df.columns) - set(track_columns))
    ]

    track_data = [track['track'] for track in tracks]

    artists = []
    for artist in [track['artists'] for track in track_data]:
        artists.append(','.join(artist['name'] for artist in artist))

    df['artist'] = artists
    df['name'] = [track['name'] for track in track_data]
    return df

def process_playlists(*playlists):
    playlist_gen = (
        playlist for playlist in sp.current_user_playlists()['items']
        if playlist['name'] in playlists
    )
    for playlist in playlist_gen:
        playlist_id, playlist_name = playlist['id'], playlist['name']
        tracks, track_data, sections, segments = get_track_data(playlist_id)
        track_feature_df = get_track_features(tracks)
        playlist_name = '_'.join(map(str.lower, playlist_name.split()))
        track_data.merge(
                track_feature_df, on=['artist', 'name']
            ).to_csv(
                f'data/{playlist_name}_tracks.csv', index=False
        )
        sections.to_csv(f'data/{playlist_name}_sections.csv', index=False)
        segments.to_csv(f'data/{playlist_name}_segments.csv', index=False)

process_playlists("AvantRock")