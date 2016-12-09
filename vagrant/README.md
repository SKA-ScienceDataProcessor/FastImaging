Pre-requisites:

You will need:

- [Virtualbox](https://www.virtualbox.org/wiki/Downloads)
- [Vagrant](https://www.vagrantup.com/downloads.html)
- [Ansible](http://docs.ansible.com/ansible/intro_installation.html)
  (Easily installed e.g. via pip, ``pip install --user ansible``)

The vagrant-cachier plugin is recommended to save on download times, you can 
install it with:

    vagrant plugin install vagrant-cachier
    
Next, grab any required ansible roles:

    ansible-galaxy install -f -r requirements.yml
    
And you can finally boot and provision the virtual machine:

    vagrant up
    
To re-run the provisioning scripts, you can either 
    
    vagrant provision
    
or you can directly run the ansible scripts - ansible knows how to ssh to the
virtual machine by looking at the `ansible.cfg` file in this directory, which 
in turn points to a vagrant-generated configuration file:

    ansible-playbook -vv provision_and_build.yml
