import librosa
import os

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
    
    return iteminfo, speakers
