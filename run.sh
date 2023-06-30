CUDA_VISIBLE_DEVICES=0 python exp_runner.py --mode train --conf ./confs/wmask_vox_24.conf --case dtu_scan24
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode train --conf ./confs/wmask_vox_24.conf --case dtu_scan37
# CUDA_VISIBLE_DEVICES=2 python exp_runner.py --mode train --conf ./confs/wmask_vox.conf --case dtu_scan40
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode train --conf ./confs/wmask_vox_24.conf --case dtu_scan55
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode train --conf ./confs/wmask_vox_24.conf --case dtu_scan63 #
# CUDA_VISIBLE_DEVICES=2 python exp_runner.py --mode train --conf ./confs/wmask_vox.conf --case dtu_scan65
# CUDA_VISIBLE_DEVICES=3 python exp_runner.py --mode train --conf ./confs/wmask_vox_24.conf --case dtu_scan69
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode train --conf ./confs/wmask_vox.conf --case dtu_scan83 #
# CUDA_VISIBLE_DEVICES=1 python exp_runner.py --mode train --conf ./confs/wmask_vox_97.conf --case dtu_scan97
# CUDA_VISIBLE_DEVICES=0 python exp_runner.py --mode train --conf ./confs/wmask_vox_24.conf --case dtu_scan105 #
# CUDA_VISIBLE_DEVICES=3 python exp_runner.py --mode train --conf ./confs/wmask_vox_24.conf --case dtu_scan106
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode train --conf ./confs/wmask_vox_24.conf --case dtu_scan110
# CUDA_VISIBLE_DEVICES=2 python exp_runner.py --mode train --conf ./confs/wmask_vox_114.conf --case dtu_scan114
# CUDA_VISIBLE_DEVICES=3 python exp_runner.py --mode train --conf ./confs/wmask_vox_24.conf --case dtu_scan118
# CUDA_VISIBLE_DEVICES=0 python exp_runner.py --mode train --conf ./confs/wmask_vox.conf --case dtu_scan122

# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode psnr --conf ./confs/wmask_vox.conf --case dtu_scan24
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode psnr --conf ./confs/wmask_vox_37.conf --case dtu_scan37
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode psnr --conf ./confs/wmask_vox.conf --case dtu_scan40
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode psnr --conf ./confs/wmask_vox.conf --case dtu_scan55
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode psnr --conf ./confs/wmask_vox.conf --case dtu_scan63
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode psnr --conf ./confs/wmask_vox.conf --case dtu_scan65
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode psnr --conf ./confs/wmask_vox.conf --case dtu_scan69
# CUDA_VISIBLE_DEVICES=3 python exp_runner.py --mode psnr --conf ./confs/wmask_vox.conf --case dtu_scan83
# CUDA_VISIBLE_DEVICES=3 python exp_runner.py --mode psnr --conf ./confs/wmask_vox.conf --case dtu_scan97
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode psnr --conf ./confs/wmask_vox.conf --case dtu_scan105
# CUDA_VISIBLE_DEVICES=3 python exp_runner.py --mode psnr --conf ./confs/wmask_vox.conf --case dtu_scan106
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode psnr --conf ./confs/wmask_vox.conf --case dtu_scan110
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode psnr --conf ./confs/wmask_vox.conf --case dtu_scan114
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode psnr --conf ./confs/wmask_vox.conf --case dtu_scan118
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode psnr --conf ./confs/wmask_vox.conf --case dtu_scan122

# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode validate_mesh --conf ./confs/wmask_vox.conf --case dtu_scan24
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode validate_mesh --conf ./confs/wmask_vox.conf --case dtu_scan37
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode validate_mesh --conf ./confs/wmask_vox_test.conf --case dtu_scan40
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode validate_mesh --conf ./confs/wmask_vox.conf --case dtu_scan55
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode validate_mesh --conf ./confs/wmask_vox.conf --case dtu_scan63
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode validate_mesh --conf ./confs/wmask_vox.conf --case dtu_scan65
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode validate_mesh --conf ./confs/wmask_vox.conf --case dtu_scan69
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode validate_mesh --conf ./confs/wmask_vox.conf --case dtu_scan83
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode validate_mesh --conf ./confs/wmask_vox.conf --case dtu_scan97
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode validate_mesh --conf ./confs/wmask_vox.conf --case dtu_scan105
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode validate_mesh --conf ./confs/wmask_vox.conf --case dtu_scan106
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode validate_mesh --conf ./confs/wmask_vox.conf --case dtu_scan110
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode validate_mesh --conf ./confs/wmask_vox.conf --case dtu_scan114
# CUDA_VISIBLE_DEVICES=4 python exp_runner.py --mode validate_mesh --conf ./confs/wmask_vox.conf --case dtu_scan118
# CUDA_VISIBLE_DEVICES=0 python exp_runner.py --mode validate_mesh --conf ./confs/wmask_vox.conf --case dtu_scan122

# python eval_mesh.py --data /home/lihai/project/NeuS/exp/dtu_scan24/wmask_vox/psnr/final.ply --scan 24 --mode mesh --dataset_dir /mnt/nas_8/datasets/dtu/ --vis_out_dir /home/lihai/project/NeuS/exp/dtu_scan24/wmask_vox/psnr/
# # python eval_mesh.py --data /home/lihai/project/NeuS/exp/dtu_scan37/wmask_vox/psnr/final.ply --scan 37 --mode mesh --dataset_dir /mnt/nas_8/datasets/dtu/ --vis_out_dir /home/lihai/project/NeuS/exp/dtu_scan37/wmask_vox/psnr/
# # python eval_mesh.py --data /home/lihai/project/NeuS/exp/dtu_scan40/wmask_vox/psnr/final.ply --scan 40 --mode mesh --dataset_dir /mnt/nas_8/datasets/dtu/ --vis_out_dir /home/lihai/project/NeuS/exp/dtu_scan40/wmask_vox/psnr/
# python eval_mesh.py --data /home/lihai/project/NeuS/exp/dtu_scan55/wmask_vox/psnr/final.ply --scan 55 --mode mesh --dataset_dir /mnt/nas_8/datasets/dtu/ --vis_out_dir /home/lihai/project/NeuS/exp/dtu_scan55/wmask_vox/psnr/
# # python eval_mesh.py --data /home/lihai/project/NeuS/exp/dtu_scan63/wmask_vox/psnr/final.ply --scan 63 --mode mesh --dataset_dir /mnt/nas_8/datasets/dtu/ --vis_out_dir /home/lihai/project/NeuS/exp/dtu_scan63/wmask_vox/psnr/
# python eval_mesh.py --data /home/lihai/project/NeuS/exp/dtu_scan65/wmask_vox/psnr/final.ply --scan 65 --mode mesh --dataset_dir /mnt/nas_8/datasets/dtu/ --vis_out_dir /home/lihai/project/NeuS/exp/dtu_scan65/wmask_vox/psnr/
# python eval_mesh.py --data /home/lihai/project/NeuS/exp/dtu_scan69/wmask_vox/psnr/final.ply --scan 69 --mode mesh --dataset_dir /mnt/nas_8/datasets/dtu/ --vis_out_dir /home/lihai/project/NeuS/exp/dtu_scan69/wmask_vox/psnr/
# # python eval_mesh.py --data /home/lihai/project/NeuS/exp/dtu_scan83/wmask_vox/psnr/final.ply --scan 83 --mode mesh --dataset_dir /mnt/nas_8/datasets/dtu/ --vis_out_dir /home/lihai/project/NeuS/exp/dtu_scan83/wmask_vox/psnr/
# # python eval_mesh.py --data /home/lihai/project/NeuS/exp/dtu_scan97/wmask_vox/psnr/final.ply --scan 97 --mode mesh --dataset_dir /mnt/nas_8/datasets/dtu/ --vis_out_dir /home/lihai/project/NeuS/exp/dtu_scan97/wmask_vox/psnr/
# python eval_mesh.py --data /home/lihai/project/NeuS/exp/dtu_scan105/wmask_vox/psnr/final.ply --scan 105 --mode mesh --dataset_dir /mnt/nas_8/datasets/dtu/ --vis_out_dir /home/lihai/project/NeuS/exp/dtu_scan105/wmask_vox/psnr/
# # python eval_mesh.py --data /home/lihai/project/NeuS/exp/dtu_scan106/wmask_vox/psnr/final.ply --scan 106 --mode mesh --dataset_dir /mnt/nas_8/datasets/dtu/ --vis_out_dir /home/lihai/project/NeuS/exp/dtu_scan106/wmask_vox/psnr/
# python eval_mesh.py --data /home/lihai/project/NeuS/exp/dtu_scan110/wmask_vox/psnr/final.ply --scan 110 --mode mesh --dataset_dir /mnt/nas_8/datasets/dtu/ --vis_out_dir /home/lihai/project/NeuS/exp/dtu_scan110/wmask_vox/psnr/
# python eval_mesh.py --data /home/lihai/project/NeuS/exp/dtu_scan114/wmask_vox/psnr/final.ply --scan 114 --mode mesh --dataset_dir /mnt/nas_8/datasets/dtu/ --vis_out_dir /home/lihai/project/NeuS/exp/dtu_scan114/wmask_vox/psnr/
# python eval_mesh.py --data /home/lihai/project/NeuS/exp/dtu_scan118/wmask_vox/psnr/final.ply --scan 118 --mode mesh --dataset_dir /mnt/nas_8/datasets/dtu/ --vis_out_dir /home/lihai/project/NeuS/exp/dtu_scan118/wmask_vox/psnr/
# # python eval_mesh.py --data /home/lihai/project/NeuS/exp/dtu_scan122/wmask_vox/psnr/final.ply --scan 122 --mode mesh --dataset_dir /mnt/nas_8/datasets/dtu/ --vis_out_dir /home/lihai/project/NeuS/exp/dtu_scan122/wmask_vox/psnr/