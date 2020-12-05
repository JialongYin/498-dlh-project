from fid_score import FidScore

real_img_dir = 'real_imgs/'
fake_img_dir = 'results/x_rays_pgan_output/'
fid = FidScore([real_img_dir, fake_img_dir], 'cpu', 429)
score = fid.calculate_fid_score()
print("fid scores:{}".format(score))
