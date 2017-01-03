tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

f_train = open('train_songs_list_final.txt','w')
f_gt_train = open('train_gt_list_final.txt','w')

f_test = open('test_songs_gtzan_list.txt','w')
f_gt_test = open('test_gt_gtzan_list.txt','w')

for index, tag in enumerate(tags):
    for i in range(0,100):
        if i<10:
            song_path = '/imatge/ajimenez/work/DSAP/music_tagger/music_dataset_mp3/gtzan/genres/'+tag+'/'+tag+'.0000'+str(i)+'.au\n'
        else:
            song_path = '/imatge/ajimenez/work/DSAP/music_tagger/music_dataset_mp3/gtzan/genres/' + tag + '/' + tag + '.000' + str(i) + '.au\n'
        print song_path
        f_test.write(song_path)
        f_gt_test.write(str(index)+'\n')


for index, tag in enumerate(tags):
    for i in range(1,21):
        song_path = '/imatge/ajimenez/work/DSAP/music_tagger/music_dataset_mp3/music_dataset/'+tag+'/Train/'+str(i)+'.mp3\n'
        print song_path
        f_train.write(song_path)
        f_gt_train.write(str(index)+'\n')
        if i < 11:
            song_path = '/imatge/ajimenez/work/DSAP/music_tagger/music_dataset_mp3/music_dataset/' + tag + '/Test/' + str(i) + '.mp3\n'
            print song_path
            f_train.write(song_path)
            f_gt_train.write(str(index) + '\n')
