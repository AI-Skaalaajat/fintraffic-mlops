# Metropolia Garage Project Template
## Repository for materials related to Metropolia Garage projects

This is a repository created to serve as a standard template for all official Metropolia Garage projects. 
In this repository, you can find the standard project folder structure, project documentation template and guide as well as introduction to how to use git and github.

**Please take the time to read this MD before you start your project!** 

### Required work for course credits

If you are doing project work for AIoT / Robo -garages as part of a course, there are few requirements that **you must fulfill:**

#### Documentation requirements

We expect you to not only properly document your project, but also to write usage instructions. This README file is a great way to easily format and visualize usage and deployment instructions for your project on the repository front page. 
If you are new to Markdown, refer to this [cheatsheet](https://github.com/adam-p/markdown-here/wiki/markdown-cheatsheet).

For the actual project documentation, there is a formatted template in the Docs folder you can use if you want, but you don't need to. The important thing is that the format and language of the documentation follows the Metropolia Thesis guidelines. 

*If you are doing your Innovation project report / Thesis, that will act as your project report for the garage! In that case, please upload it with the project files or link to it in theseus.*

The report should include atleast the following:

- What was the project about?
- What are the things you did?
- What were the problems you faced and how did you solve them?
- What work remains to be done or what issues need to be fixed?

And finally, list the tools/software you used for the project and provide brief description of what each of them are for.

You are the expert of your own project and it is critical that the work and innovation you have spent your time on is easily available to us in the future! 


### How to proceed
If you don't have Git installed yet, refer to [setting up Git](https://docs.github.com/en/get-started/git-basics/set-up-git).

We highly recommend you to start with the amazing [First Contributions](https://github.com/firstcontributions/first-contributions) guide if you are new to GitHub. 

This template is made with forking and pull requests in mind to systemize workflow and make it easy for Metropolia Staff to access your project later, while allowing you to work as freely as possible.

#### Expected Workflow:
![Flowchart for project](project_flow.png)

### Project structure

In this project template, you can find a readymade folder structure. Your project may not need some of the folders, in which case please delete them. 
**Otherwise, use the provided structure.**

#### Overview of the folders

##### Deployment
* Any files or general dependencies that are necessary to get the project running

##### Docs
* Project Report, Notes, Guides, you can find the Word Documentation Template here with basic formatting made

##### Firmware
* Files and software, which are necessary to get your hardware running

##### Hardware
* Schematics, parts and other things related to your project hardware

##### Media
*  All media material related to the project, sorted

##### Presentation
*  Your powerpoint presentations, if you have any

##### References
*  Scientific papers, datasheets, web links - Please name them in a clear manner

##### Software
*  Your actual code, models, libraries etc.

#### Example project structure

1.  Project-Name/   

 2. │   

 3. ├──  docs/            # Documentation and reports   

 4. │   ├── project-report.docx   # Main project report   

 5. │   ├── requirements.txt  	# Project requirements & specs     

 9. │      

11. ├──  software/    # AI, IoT, and application code
24. │   ├──  deployment/ # Deployment files, Docker, cloud configurations   

25. │   │   ├── docker/   

26. │   │   ├── k8s/   

27. │   │   ├── cloud-setup/       

14. │   ├── models/      # AI/ML models and training scripts   

15. │   ├── code/     # Project source code, helper scripts, automation   

16. │   ├── tests/       # Unit tests, integration tests   

17. │   ├── README.md    # Explanation of software structure   

18. │   │   

19. │   ├──  firmware/  # Microcontroller firmware (if any)   

20. │   │   ├── esp32/   

21. │   │   ├── jetson/   

22. │   │   ├── raspberry-pi/   

28. │   

29. ├──  hardware/      # Any hardware-related files   

30. │   ├── schematics/      # Circuit diagrams, PCB designs   

31. │   ├── parts-list.xlsx    # List of components and BOM   

32. │   ├── assembly-guide.md  # Instructions for assembling hardware   

33. │   

34. ├──  media/        # Images, videos, marketing materials   

35. │   ├──  photos/         # Pictures of the project   

36. │   ├──  videos/         # Any demo or tutorial videos   

37. │   ├──  graphics/       # Logos, banners, presentation materials   

38. │   

39. ├──  presentations/       # Slides, posters, marketing materials   

40. │   ├── presentation.pptx   

41. │   ├── poster.pdf   

42. │   

43. ├──  references/          # Papers, useful links, background materials   

44. │   ├── papers/   

45. │   ├── links.md   

46. │   

47. ├── .gitignore              # Ignore unnecessary files for version control   

48. ├── README.md               # General project overview   

49. └── LICENSE                 # Open-source license, if applicable  

50.   


### Using Git

#### Getting started

**Fork this repository**: Click the "Fork" button in the top-right corner to create your own copy.
Then set it up locally, using the CMD line:
>git clone [fork-URL]

(Downloads a copy of your fork to your local machine)

>cd [repository-name]          

(Moves into the project directory)

>git remote add upstream [project-repository-URL] 

(Links Git to the original repository (This one))

#### Development Workflow

**Create a feature branch**:

>git checkout -b [new-branch-name]

(Creates and switches to a new branch)

**Make your changes** and commit regularly with clear messages:
>git add . 

(Stages all modified files for commit)

>git commit -m "Implement feature X"

(Records changes with a message)

**Merge to your fork's main branch** when you have a stable checkpoint you want to save

>git checkout main                   

(Switch back to your main branch)

>git merge [new-branch-name]

(Integrate your feature into main)

>git push origin main                

(Push the updated main to your forked repository)

Generally, you want to keep your local main branch as a functional backup and push any feature you are actively working on into its own separate branch. The main branch will be the pull point for the real project repository.

#### Submitting your work to us

**Create a pull request**:
* Go to this repository
* Click "Pull requests" > "New pull request"
* Click "compare across forks"
* Select your fork's main branch as the head repository
* Click "Create pull request"
* Add a title and description explaining your changes

We will accept your changes and they will be copied to this repository. 
