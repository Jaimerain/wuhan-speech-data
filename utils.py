import librosa
import os
import numpy as np

def extract_metadata(directory):
    """extract metadata from the data filenames
       normalise some values to give
        gender: m or f
        language: ch or en
        item: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (as a string)
        speaker: a unique speaker id (string)
    """
    iteminfo = {}
    for filename in librosa.util.find_files(directory):
        basename, ext = os.path.splitext(os.path.basename(filename))
        basename = basename.replace('6_gb', '6gb')
        basename = basename.replace('_','-').replace('--', '-')
        sp, gender, lang, item = basename.split('-')
        lang = lang.lower()
        if lang == 'cn':
            lang = 'ch'
        if gender == 'W':
            gender = 'f'
        gender = gender.lower()
        if item[0] == '0':
            item = item[1]
        iteminfo[filename] = {
                'speaker': sp,
                'gender': gender,
                'language': lang,
                'item': item
                }

    speakers = set([iteminfo[i]['speaker'] for i in iteminfo])

    return iteminfo, list(speakers)


def get_data_labels(iteminfo, filenames, keyword):
    """Given the iteminfo dictionary and a list of filenames,
     return a list of the matching property values for the
     given keywords.
     eg. get_data_labels(iteminfo, filenames, 'gender') will
     return a list of 'm' or 'f' for each file"""

    return np.array([iteminfo[filename][keyword] for filename in filenames])


def get_data_for(iteminfo, keyword, value):
    """Given the iteminfo dictionary return a list of
    files where keyword=value in the metadata, eg
    get_data_for(iteminfo, 'gender', 'm') will
    return all files for male speakers"""

    result = []
    for filename in iteminfo:
        if keyword in iteminfo[filename]:
            kwvalue = iteminfo[filename][keyword]
            if kwvalue == value or kwvalue in value:
                result.append(filename)
    return np.array(result)

def features(datafile):
    y, sr = librosa.load(datafile)
    mfcc = librosa.feature.mfcc(y, sr = sr)
    return mfcc.mean(axis=1)

#def features(datafile, label):
#    """Generate mfcc feature vectors for a single data file
#    return a 2 dimensional numpy array with one row per frame
#    and an array of labels for each frame"""
#    y, sr = librosa.load(datafile)
#    mfcc = librosa.feature.mfcc(y=y, sr=sr)
#    labels = np.full((mfcc.shape[1],), label)
#    return np.array(mfcc), labels

def extract_frame_features(datafiles, target):
    """Given a list of sound files and their target labels
    Compute a sequence of features for each file
    Concatenate these features into a single np.array
    Return these features and a corresponding array
    of target labels, one for every frame"""

    data = None
    frame_target = []
    for i in range(len(target)):
        frames, labels = features(datafiles[i], target[i])
        frame_target.extend(labels)
        if data is None:
            # transpose the frames
            data = frames.T
        else:
            # concatenate the tranposed frames
            data = np.concatenate((data, frames.T))

    frame_target = np.array(frame_target)

    return data, frame_target

