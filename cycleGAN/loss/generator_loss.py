import torch

def get_adversarial_loss(real_content, real_style, disc_content, disc_style, gen_c2s, gen_s2c, adversarial_criterion):
    fake_content = gen_s2c(real_style)
    disc_fake_content_hat = disc_content(fake_content)
    adversarial_loss_s2c = adversarial_criterion(disc_fake_content_hat, torch.ones_like(disc_fake_content_hat))

    fake_style = gen_c2s(real_content)
    disc_fake_style_hat = disc_style(fake_style)
    adversarial_loss_c2s = adversarial_criterion(disc_fake_style_hat, torch.ones_like(disc_fake_style_hat))

    gen_adversarial_loss = adversarial_loss_s2c + adversarial_loss_c2s
    return gen_adversarial_loss, fake_content, fake_style

def get_identity_loss(real_content, real_style, gen_c2s, gen_s2c, identity_criterion):
    identity_content = gen_s2c(real_content)
    content_identity_loss = identity_criterion(identity_content, real_content)

    identity_style = gen_c2s(real_style)
    style_identity_loss = identity_criterion(identity_style, real_style)

    return content_identity_loss + style_identity_loss

def get_cycle_consistency_loss(real_content, real_style, fake_content, fake_style, gen_c2s, gen_s2c, cycle_criterion):
    cycle_content = gen_s2c(fake_style)
    content_cycle_loss = cycle_criterion(cycle_content, real_content)

    cycle_style = gen_c2s(fake_content)
    style_cycle_loss = cycle_criterion(cycle_style, real_style)
    
    return content_cycle_loss + style_cycle_loss

def get_gen_loss(real_content, real_style, gen_c2s, gen_s2c, disc_content, disc_style, adv_criterion, identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10):
    adversarial_loss, fake_content, fake_style = get_adversarial_loss(real_content, real_style, disc_content, disc_style, gen_c2s, gen_s2c, adv_criterion)
    identity_loss = get_identity_loss(real_content, real_style, gen_c2s, gen_s2c, identity_criterion)
    cycle_loss = get_cycle_consistency_loss(real_content, real_style, fake_content, fake_style, gen_c2s, gen_s2c, cycle_criterion)

    gen_loss = adversarial_loss + lambda_identity * identity_loss + lambda_cycle * cycle_loss
    return gen_loss, fake_content, fake_style