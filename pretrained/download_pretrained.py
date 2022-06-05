import wget
from tqdm import tqdm

project_url = 'https://github.com/FanChiMao/Competition-2022-Pytorch-Orchid_Classification'

def main():
    print('It will cost about 10-15 minutes to download...')
    with tqdm(total=11) as bar:
        wget.download(project_url + '/releases/download/v0.0/convnext.pth')
        bar.update(1) #1
        wget.download(project_url + '/releases/download/v0.0/beit_1.pth')
        bar.update(1) #2 
        wget.download(project_url + '/releases/download/v0.0/beit_2.pth')
        bar.update(1) #3
        wget.download(project_url + '/releases/download/v0.0/swin.pth')
        bar.update(1) #4
        wget.download(project_url + '/releases/download/v0.0/vit.pth')
        bar.update(1) #5
        wget.download(project_url + '/releases/download/v0.0/dmnfnet.pth')
        bar.update(1) #6
        wget.download(project_url + '/releases/download/v0.0/ecaresnet_50.pth')
        bar.update(1) #7
        wget.download(project_url + '/releases/download/v0.0/efficient.pth')
        bar.update(1) #8
        wget.download(project_url + '/releases/download/v0.0/regnet.pth')
        bar.update(1) #9
        wget.download(project_url + '/releases/download/v0.0/volo.pth')
        bar.update(1) #10
        wget.download(project_url + '/releases/download/v0.0/resmlp.pth')
        bar.update(1) #11
        wget.download(project_url + '/releases/download/v0.0/')
        bar.update(1) #12
        wget.download(project_url + '/releases/download/v0.0/')
        bar.update(1) #13

if __name__ == '__main__':
    print(f'Start downloading pretrained models from {project_url}/releases/tag/v0.0')
    main()
    print('Done !!')
