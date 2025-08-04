import glob
import os

# 00010*_4k_v2_all_checkpoints/*/*bba*.png

proteins = ['bba', 'chignolin', 'trpcage', 'homeodomain', 'proteinb', 'wwdomain']
for protein in proteins:
    files = glob.glob(f'sims/00010*_4k_v2_all_checkpoints/*/*{protein}*.png')
    for f in files:
        ch = int(f.split('/')[-2].split('-')[-1])
        print('f', f)
        print('ch', ch)
        cmd= f'cp {f} sims/all/{protein}_checkpoint%02d.png' % ch
        print(cmd)
        os.system(cmd)
        # write checkpoint number as text title onto the pngs
        cmd = f'convert sims/all/{protein}_checkpoint%02d.png -pointsize 30 -fill white -annotate +50+50 "checkpoint {ch}" sims/all/{protein}_checkpoint%02d.png' % (ch, ch)
        os.system('cmd')