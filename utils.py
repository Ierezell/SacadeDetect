from settings import (DEVICE, PATH_WEIGHTS_RNN,
                      PATH_WEIGHTS_CLASSIFIER, PATH_WEIGHTS_AUTOENCODER)
from archi import SacadeRnn, Classifier, Autoencoder
import torch


def load_models(nb_pers, load_previous_state, load_classifier=True):
    sacade_rnn = SacadeRnn(nb_pers)
    classifier = Classifier(nb_pers)
    autoencoder = Autoencoder(64, 64)
    if load_previous_state:
        sacade_rnn.load_state_dict(torch.load(PATH_WEIGHTS_RNN))
        autoencoder.load_state_dict(torch.load(PATH_WEIGHTS_AUTOENCODER))
        if load_classifier:
            classifier.load_state_dict(torch.load(PATH_WEIGHTS_CLASSIFIER))
    sacade_rnn = sacade_rnn.to(DEVICE)
    classifier = classifier.to(DEVICE)
    autoencoder = autoencoder.to(DEVICE)
    return sacade_rnn, classifier, autoencoder


class Checkpoints:
    def __init__(self):
        self.losses = {"loss": []}
        self.best_loss = 1e10

    def addCheckpoint(self, model, loss):
        loss = loss.detach()
        self.losses[model].append(loss)

    def save(self, model, loss, sacade_rnn, classifier, autoencoder):
        if len(self.losses[model]) % 500 == 0:
            print('\n'+'-'*21+"\n| Poids sauvegardés |\n"+'-'*21+'\n')
            self.best_loss = loss
            torch.save(sacade_rnn.state_dict(), PATH_WEIGHTS_RNN)
            torch.save(classifier.state_dict(), PATH_WEIGHTS_CLASSIFIER)
            torch.save(autoencoder.state_dict(), PATH_WEIGHTS_AUTOENCODER)

    def visualize(self, fig, axes,
                  gt_landmarks, synth_im, gt_im, *models,
                  save_fig=False, name='plop'):
        "-----------------------"
        # TODO Faire une vraie accuracy
        accuracy = 0.5
        "------------------------"
        # plt.figure('Mon')
        # plt.clf()
        im_landmarks = gt_landmarks.detach()[0].cpu().permute(1, 2, 0).numpy()
        im_synth = synth_im.detach()[0].cpu().permute(1, 2, 0).numpy()
        im_gt = gt_im.detach()[0].cpu().permute(1, 2, 0).numpy()
        # print("IMAGES : ")
        # print(im_gt.min(), im_gt.max())
        # print(im_synth.min(), im_synth.max())
        # print(im_landmarks.min(), im_landmarks.max())
        axes[0, 0].clear()
        axes[0, 0].imshow(im_landmarks/im_landmarks.max())
        axes[0, 0].axis("off")
        axes[0, 0].set_title('Landmarks')

        axes[0, 1].clear()
        axes[0, 1].imshow(im_synth/im_synth.max())
        axes[0, 1].axis("off")
        axes[0, 1].set_title('Synthesized image')

        axes[0, 2].clear()
        axes[0, 2].imshow(im_gt/im_gt.max())
        axes[0, 2].axis("off")
        axes[0, 2].set_title('Ground truth')

        axes[1, 0].clear()
        axes[1, 0].plot(self.losses["dsc"], label='Disc loss')
        axes[1, 0].set_title('Disc loss')

        axes[1, 1].clear()
        axes[1, 1].plot(self.losses["adv"], label='Adv loss')
        axes[1, 1].plot(self.losses["mch"], label='Mch loss')
        axes[1, 1].plot(self.losses["cnt"], label='Cnt loss')
        # axes[1, 1].plot(np.array(self.losses["adv"]) +
        #                 np.array(self.losses["mch"]) +
        #                 np.array(self.losses["cnt"]), label='EmbGen loss')
        axes[1, 1].set_title('EmbGen losses')
        axes[1, 1].legend()

        axes[1, 2].clear()
        axes[1, 2].plot(accuracy)
        axes[1, 2].set_title('Accuracy')

        for i, m in enumerate(models):
            ave_grads = []
            max_grads = []
            layers = []
            for n, p in m.named_parameters():
                if(p.requires_grad) and ("bias" not in n):
                    layers.append('.'.join(n.split('.')[: -1]))
                    try:
                        gradient = p.grad.cpu().detach()
                        ave_grads.append(gradient.abs().mean())
                        max_grads.append(gradient.abs().max())
                    except AttributeError:
                        ave_grads.append(0)
                        max_grads.append(0)
            axes[2, i].clear()
            axes[2, i].bar(np.arange(len(max_grads)), max_grads,
                           alpha=0.5, lw=1, color="c")
            axes[2, i].bar(np.arange(len(ave_grads)), ave_grads,
                           alpha=0.7, lw=1, color="r")
            axes[2, i].hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
            axes[2, i].set_xticks(np.arange(len(layers)))
            axes[2, i].set_xticklabels(layers, rotation="vertical",
                                       fontsize='small')
            axes[2, i].set_xlim(left=0, right=len(ave_grads))
            axes[2, i].set_ylim(bottom=0, top=max(ave_grads)+1)
            # zoom in on the lower gradient regions
            axes[2, i].set_xlabel("Layers")
            axes[2, i].set_ylabel("average gradient")
            axes[2, i].set_title(f"{m.__class__.__name__} gradient flow")
            axes[2, i].grid(True)
            axes[2, i].legend([Line2D([0], [0], color="c", lw=4),
                               Line2D([0], [0], color="r", lw=4)],
                              ['max-gradient', 'mean-gradient'])
        if save_fig:
            fig.savefig(f"{ROOT_IMAGE}{name}.png", dpi=fig.dpi)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()


# if iteration % PRINT_EVERY == 0:
#             print_loss_avg = print_loss / PRINT_EVERY
#             print(f"Iteration: {iteration};  ",
#                   f"Percent complete: {(iteration / N_ITERATION * 100):.1f}% ",
#                   f"Average loss: {print_loss_avg:.4f}")
#             print_loss = 0

#         # Save checkpoint
#         if (iteration % SAVE_EVERY == 0):
#             directory = os.path.join(SAVE_DIR, MODEL_NAME,
#                                      '{}-{}_{}'.format(ENCODER_N_LAYERS,
#                                                        DECODER_N_LAYERS,
#                                                        HIDDEN_SIZE))
#             if not os.path.exists(directory):
#                 os.makedirs(directory)
#             torch.save({
#                 'iteration': iteration,
#                 'en': encoder.state_dict(),
#                 'de': decoder.state_dict(),
#                 'en_opt': encoder_optimizer.state_dict(),
#                 'de_opt': decoder_optimizer.state_dict(),
#                 'loss': loss,
#                 'voc_dict': voc.__dict__,
#                 'embedding': embedding.state_dict()
#             }, os.path.join(directory, f'{iteration}_checkpoint.tar'))


# encoder_sd = checkpoint['en']
# decoder_sd = checkpoint['de']
