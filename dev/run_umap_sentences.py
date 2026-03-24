"""UMAP of sentence-level embeddings from the pre-1900 base model."""
import os, sys, json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys, _patch_missing_keys
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer

matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['figure.facecolor'] = 'white'

FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'analysis')
os.makedirs(FIGDIR, exist_ok=True)

# ---- Load model ----
print("=== Loading model ===")
CACHE_DIR = os.path.expanduser("~/hf_cache")
local_dir = os.path.join(CACHE_DIR, "d34-22btok")
tok_dir = os.path.join(local_dir, "tokenizer")
tokenizer = RustBPETokenizer.from_directory(tok_dir)

device = torch.device("cuda")
model_data, _, meta_data = load_checkpoint(local_dir, 10507, device, load_optimizer=False)
model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
cfg = meta_data["model_config"]
_patch_missing_config_keys(cfg)
model_config = GPTConfig(**cfg)
_patch_missing_keys(model_data, model_config)
with torch.device("meta"):
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()
model.load_state_dict(model_data, strict=True, assign=True)
model.bfloat16()
model.eval()
print(f"Model: {model_config.n_layer}L, {model_config.n_embd}d")
del model_data

# ---- Hook for final layer ----
final_hidden = {}
hook = list(model.transformer.h)[-1].register_forward_hook(
    lambda m, i, o: final_hidden.update({'h': o.detach()}))

@torch.no_grad()
def get_sentence_embedding(text):
    bos = tokenizer.get_bos_token_id()
    tokens = tokenizer.encode(text, prepend=bos)
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    _ = model(ids)
    return final_hidden['h'][0, -1, :].float().cpu().numpy()

# ---- 200 sentences per category ----
SENTENCES = {
    "Sciences": [
        # Physics & mechanics
        "The force of gravity draws all bodies toward the centre of the earth in proportion to their mass.",
        "A body in motion continues in a straight line unless acted upon by an external force.",
        "The acceleration of a falling body near the surface of the earth is approximately constant.",
        "The period of a simple pendulum depends upon its length and the strength of gravity.",
        "A lever amplifies force at the expense of the distance through which it acts.",
        "The principle of the inclined plane reduces the force required to raise a heavy weight.",
        "Friction between surfaces converts the energy of motion into heat.",
        "The momentum of a body is the product of its mass and its velocity.",
        "An elastic collision preserves both momentum and kinetic energy.",
        "The centre of gravity of a uniform body lies at its geometric centre.",
        "Pressure in a fluid increases with depth according to the weight of the overlying column.",
        "Archimedes showed that a body immersed in fluid is buoyed up by a force equal to the weight of fluid displaced.",
        "The speed of sound in air depends upon the temperature and the density of the atmosphere.",
        "A vibrating string produces a musical note whose pitch depends upon its length and tension.",
        "The laws of the pendulum were first studied systematically by Galileo.",
        "Torricelli demonstrated that the atmosphere exerts a pressure sufficient to support a column of mercury.",
        "Boyle showed that the volume of a gas varies inversely with the pressure applied to it.",
        "The specific gravity of a substance is the ratio of its density to that of water.",
        "Hooke observed that the extension of a spring is proportional to the force applied.",
        "The motion of a projectile follows a parabolic path under the influence of gravity alone.",
        # Optics & light
        "Newton demonstrated with a glass prism that white light is composed of rays of every colour.",
        "The angle of incidence equals the angle of reflection when light strikes a polished surface.",
        "A convex lens brings parallel rays of light to a focus at a single point.",
        "The rainbow is produced by the refraction and internal reflection of sunlight within drops of rain.",
        "Young showed by his double-slit experiment that light behaves as a wave.",
        "The speed of light was first estimated by observations of the moons of Jupiter.",
        "A concave mirror gathers light and forms a real image at its focal point.",
        "The index of refraction of glass is greater than that of air or water.",
        "Polarisation of light proves that it is a transverse wave rather than a longitudinal one.",
        "Diffraction of light around the edge of an obstacle produces coloured fringes.",
        "Huygens proposed that light propagates as a wave through a subtle medium filling all space.",
        "The spectrum of sunlight contains dark lines corresponding to the absorption of certain wavelengths.",
        "Fresnel developed a mathematical theory of diffraction that accounted for the wave nature of light.",
        "The velocity of light in water is less than its velocity in air.",
        "A telescope with a larger aperture resolves finer details of distant objects.",
        # Electricity & magnetism
        "An electric current flowing through a wire produces a magnetic field around it.",
        "A compass needle aligns itself with the magnetic field of the earth.",
        "Faraday discovered that a changing magnetic field induces an electric current in a nearby conductor.",
        "The resistance of a wire increases with its length and decreases with its cross-sectional area.",
        "A voltaic pile produces a steady current of electricity by chemical action.",
        "Coulomb showed that the force between two electric charges varies as the inverse square of the distance.",
        "Lightning is a discharge of atmospheric electricity between a cloud and the ground.",
        "An electromagnet is formed by winding a coil of wire around an iron core.",
        "The electric telegraph transmits messages over long distances by means of coded electrical signals.",
        "Ohm established that the current through a conductor is proportional to the applied voltage.",
        "Ampere showed that two parallel wires carrying current exert a force upon each other.",
        "A galvanometer detects the presence and direction of an electric current.",
        "Static electricity can be produced by rubbing glass with silk or amber with fur.",
        "The Leyden jar stores electric charge on the inner and outer surfaces of a glass vessel.",
        "Maxwell demonstrated mathematically that light is an electromagnetic wave.",
        # Thermodynamics & heat
        "Heat flows spontaneously from a hotter body to a cooler one until they reach equilibrium.",
        "The steam engine converts the energy of heated steam into mechanical work.",
        "Rumford showed by his cannon-boring experiments that heat is a form of motion rather than a substance.",
        "Carnot proved that no heat engine can be more efficient than a reversible one operating between the same temperatures.",
        "The absolute zero of temperature is the point at which all molecular motion ceases.",
        "A thermometer measures temperature by the expansion of mercury or alcohol in a glass tube.",
        "The latent heat of fusion is the energy required to melt a solid without changing its temperature.",
        "Joule established the mechanical equivalent of heat by stirring water with a paddle wheel.",
        "Conduction transfers heat through a solid by the vibration of its particles.",
        "Radiation carries heat across empty space without the need for any intervening medium.",
        # Chemistry
        "Lavoisier demonstrated that combustion is the combination of a substance with oxygen from the air.",
        "Water is composed of two parts hydrogen and one part oxygen by volume.",
        "An acid reacts with a base to produce a salt and water.",
        "Dalton proposed that all matter is composed of indivisible atoms of definite weight.",
        "The periodic table arranges the elements according to their atomic weights and chemical properties.",
        "Electrolysis decomposes a compound by passing an electric current through its solution.",
        "Iron rusts when exposed to moist air through a slow process of oxidation.",
        "A catalyst accelerates a chemical reaction without being consumed in the process.",
        "The law of definite proportions states that a compound always contains the same elements in the same ratio by weight.",
        "Distillation separates the components of a liquid mixture by their different boiling points.",
        "Carbon forms the backbone of all organic compounds found in living matter.",
        "The noble gases were discovered at the end of the century and placed in a new column of the periodic table.",
        "Davy isolated potassium and sodium by the electrolysis of their molten hydroxides.",
        "Sulphuric acid is among the most important chemicals used in industry and manufacture.",
        "Phosphorus was first isolated from urine by the alchemist Hennig Brand.",
        # Biology & natural history
        "Darwin proposed that new species arise through the natural selection of favourable variations.",
        "The cell is the fundamental unit of structure and function in all living organisms.",
        "Pasteur demonstrated that fermentation is caused by living microorganisms rather than by spontaneous generation.",
        "The circulation of the blood was described by Harvey, who showed that the heart acts as a pump.",
        "Linnaeus devised a system of classification that groups organisms by genus and species.",
        "Mendel discovered the laws of heredity by crossing pea plants and counting the offspring.",
        "Fossils preserved in sedimentary rock reveal the history of life on earth.",
        "The theory of evolution explains the diversity of living forms by descent with modification.",
        "Bacteria were first observed under the microscope by Leeuwenhoek in the seventeenth century.",
        "Jenner showed that inoculation with cowpox material protects against the more dangerous smallpox.",
        "The embryos of different vertebrate species resemble one another in their early stages of development.",
        "Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
        "The nervous system transmits signals from the sense organs to the brain and from the brain to the muscles.",
        "Comparative anatomy reveals the common plan of structure underlying the diversity of animal forms.",
        "The struggle for existence ensures that only the best-adapted individuals survive to reproduce.",
        # Astronomy & geology
        "The planets revolve around the sun in elliptical orbits according to the laws of Kepler.",
        "The tides are caused by the gravitational attraction of the moon and, to a lesser extent, the sun.",
        "A solar eclipse occurs when the moon passes between the earth and the sun.",
        "The Milky Way is a vast collection of stars stretching across the night sky.",
        "Comets travel in elongated orbits and develop bright tails as they approach the sun.",
        "The age of the earth is far greater than the few thousand years suggested by a literal reading of scripture.",
        "Volcanoes erupt when molten rock from deep within the earth forces its way to the surface.",
        "Glaciers carved out the valleys and deposited the moraines that mark their former extent.",
        "Sedimentary rocks are formed by the gradual accumulation of sand, mud, and shells on the sea floor.",
        "The strata of the earth record a succession of ages, each with its own characteristic fossils.",
        "Lyell argued that the same geological processes operating today have shaped the earth over vast periods.",
        "The nebular hypothesis proposes that the solar system formed from a rotating cloud of gas and dust.",
        "Meteorites that fall to earth provide samples of material from elsewhere in the solar system.",
        "The parallax of nearby stars can be measured to determine their distance from the earth.",
        "Halley predicted the return of the comet that now bears his name by calculating its orbit.",
        # Mathematics
        "Euclid set out the foundations of geometry in his Elements, beginning from a small set of axioms.",
        "Newton and Leibniz independently developed the calculus for computing rates of change and areas.",
        "The fundamental theorem of algebra states that every polynomial equation has at least one root.",
        "Euler discovered a remarkable formula connecting the exponential function with trigonometry.",
        "Gauss proved the law of quadratic reciprocity and made profound contributions to number theory.",
        "The theory of probability originated in the correspondence of Pascal and Fermat.",
        "Fourier showed that any periodic function can be represented as a sum of sines and cosines.",
        "The concept of a limit is the foundation upon which the rigorous theory of calculus is built.",
        "A differential equation relates a function to its derivatives and arises throughout natural philosophy.",
        "The method of least squares finds the best fit of a curve to a set of observed data.",
        # Extra physics/science
        "The conservation of energy states that energy can be neither created nor destroyed.",
        "A wave carries energy from one place to another without transporting matter.",
        "The kinetic theory of gases explains pressure and temperature in terms of molecular motion.",
        "Bernoulli showed that the pressure of a moving fluid decreases as its velocity increases.",
        "The principle of superposition states that waves passing through the same point add together.",
        "Charles showed that the volume of a gas at constant pressure is proportional to its temperature.",
        "Gay-Lussac observed that gases combine in simple ratios by volume.",
        "Avogadro proposed that equal volumes of gas at the same temperature and pressure contain equal numbers of molecules.",
        "The phenomenon of resonance occurs when a system is driven at its natural frequency.",
        "The conservation of momentum holds in all collisions between bodies.",
        "Stokes derived the law governing the slow motion of a sphere through a viscous fluid.",
        "The index of refraction of a substance determines the speed of light within it.",
        "The dispersion of light by a prism separates it into its component colours.",
        "Young measured the wavelength of light by observing the spacing of interference fringes.",
        "The specific heat of a substance is the amount of heat required to raise one gram by one degree.",
        "The law of universal gravitation states that every mass attracts every other mass.",
        "Herschel discovered infrared radiation by placing a thermometer beyond the red end of the spectrum.",
        "The piezoelectric effect produces an electric charge when certain crystals are compressed.",
        "The barometer measures atmospheric pressure by the height of a column of mercury.",
        "The hygrometer measures the humidity of the air by comparing wet and dry thermometer readings.",
    ],
    "Humanities": [
        # Philosophy — epistemology & metaphysics
        "Descartes held that the only thing that cannot be doubted is the existence of the thinking self.",
        "Locke argued that the mind begins as a blank slate and all ideas come from experience.",
        "Hume maintained that our belief in causation rests on habit rather than on reason.",
        "Kant proposed that the mind imposes categories of understanding upon the raw data of experience.",
        "Berkeley denied the existence of matter and held that to be is to be perceived.",
        "Spinoza identified God with Nature and argued that everything follows from a single substance.",
        "Leibniz proposed that the world is composed of simple substances called monads.",
        "Aristotle held that knowledge begins with the senses and proceeds by abstraction to universal truths.",
        "Plato argued that the visible world is a shadow of a higher realm of eternal forms.",
        "The problem of induction asks how we can justify general conclusions drawn from particular observations.",
        "Empiricists hold that all knowledge derives from sensory experience rather than from innate ideas.",
        "Rationalists maintain that certain truths can be known through reason alone, independent of experience.",
        "The distinction between analytic and synthetic judgments is central to the philosophy of Kant.",
        "Schopenhauer argued that the world as we experience it is a representation shaped by the will.",
        "Mill defended the principle of utility as the foundation of moral reasoning.",
        "The categorical imperative commands us to act only according to rules we could will to be universal laws.",
        "Hegel described the progress of thought as a dialectical movement from thesis through antithesis to synthesis.",
        "The Scottish school of common sense philosophy rejected the sceptical conclusions of Hume.",
        "Bacon urged that knowledge should be advanced by systematic observation and experiment rather than by syllogism.",
        "The pragmatists held that the truth of an idea consists in its practical consequences.",
        # Philosophy — ethics & political
        "Bentham proposed that the greatest happiness of the greatest number is the measure of right and wrong.",
        "Rousseau argued that man is born free but is everywhere in chains imposed by society.",
        "Hobbes described the natural state of mankind as a war of all against all.",
        "Locke defended the natural rights of life, liberty, and property against arbitrary government.",
        "The social contract theory holds that the authority of the state rests on the consent of the governed.",
        "Burke opposed the French Revolution and defended the wisdom of inherited institutions.",
        "Paine argued that the rights of man are universal and that monarchy is an affront to reason.",
        "Tocqueville observed that democracy in America fostered equality but threatened the tyranny of the majority.",
        "Montesquieu advocated the separation of powers as a safeguard against despotism.",
        "Wollstonecraft argued that women deserve the same education and rights as men.",
        "The stoics taught that virtue consists in living in accordance with reason and accepting what fate brings.",
        "Epicurus held that pleasure is the highest good but that the greatest pleasure is the absence of pain.",
        "The utilitarians sought to reform law and government on the principle of the greatest good.",
        "Adam Smith argued that the wealth of nations depends on the division of labour and free exchange.",
        "Malthus warned that population growth would inevitably outstrip the supply of food.",
        "Ricardo developed the theory of comparative advantage to explain the benefits of international trade.",
        "The abolitionists argued on moral grounds that slavery is a violation of natural human rights.",
        "The natural law tradition holds that certain moral principles are discoverable by reason alone.",
        "Machiavelli counselled the prince to be both feared and loved but above all to be effective.",
        "Voltaire championed tolerance and the freedom of expression against religious and political persecution.",
        # Religion & theology
        "The argument from design infers the existence of God from the order and complexity of nature.",
        "The problem of evil asks how a benevolent and omnipotent God can permit suffering in the world.",
        "Calvin taught the doctrine of predestination, holding that God has chosen the elect from eternity.",
        "Luther challenged the authority of the Pope and insisted that salvation comes through faith alone.",
        "The scholastic philosophers sought to reconcile the teachings of Aristotle with Christian doctrine.",
        "Aquinas offered five proofs for the existence of God based on motion, causation, and design.",
        "The deists held that God created the world but does not intervene in its affairs.",
        "The higher criticism applied the methods of historical scholarship to the study of the Bible.",
        "Missionary societies sent preachers to every continent to spread the Christian gospel.",
        "The Oxford Movement sought to revive the Catholic traditions within the Church of England.",
        "Pascal argued that it is rational to wager on the existence of God given the stakes involved.",
        "The doctrine of original sin holds that all humanity inherits the guilt of Adam's transgression.",
        "The Reformation divided Western Christendom into Catholic and Protestant confessions.",
        "Schleiermacher defined religion as the feeling of absolute dependence upon the infinite.",
        "The Quakers rejected all formal creeds and insisted on the inner light of conscience.",
        "The revivalist preachers of the Great Awakening stirred intense religious feeling across the colonies.",
        "The Jesuits established schools and missions throughout Europe, Asia, and the Americas.",
        "Natural theology argues that the existence of God can be demonstrated from the evidence of nature alone.",
        "Kierkegaard insisted that authentic faith requires a leap beyond the certainties of reason.",
        "The conflict between science and religion intensified after the publication of the theory of evolution.",
        # Literature
        "Shakespeare explored the depths of human ambition, jealousy, and love in his great tragedies.",
        "Milton composed Paradise Lost to justify the ways of God to man in heroic blank verse.",
        "Wordsworth celebrated the beauty of the natural world and the power of memory in his poetry.",
        "Dickens used the serial novel to portray the suffering of the poor in industrial England.",
        "Austen examined the manners and marriages of the English gentry with wit and precision.",
        "Goethe's Faust tells the story of a scholar who sells his soul in exchange for knowledge and experience.",
        "Tolstoy depicted the upheaval of war and the search for meaning in the lives of the Russian aristocracy.",
        "Byron scandalised and thrilled his readers with tales of passionate heroes in exotic settings.",
        "Shelley wrote odes to the west wind and the skylark that celebrate the power of the imagination.",
        "The Romantic poets rejected the formal conventions of the previous age in favour of feeling and nature.",
        "The Gothic novel combined horror, mystery, and the supernatural in decaying castles and wild landscapes.",
        "Cervantes created in Don Quixote a comic masterpiece about a knight who mistakes windmills for giants.",
        "Hugo's novels championed the cause of the poor and oppressed in the streets of Paris.",
        "Balzac aimed to catalogue the whole of French society in his vast series of interconnected novels.",
        "The epistolary novel tells its story through an exchange of letters between the characters.",
        "Keats wrote odes of extraordinary beauty to a nightingale, a Grecian urn, and the season of autumn.",
        "Coleridge composed The Rime of the Ancient Mariner, a tale of guilt and redemption on the open sea.",
        "Emerson called for a distinctly American literature free from dependence on European models.",
        "Whitman celebrated democracy and the common people in his free-verse collection Leaves of Grass.",
        "Eliot depicted the moral struggles of provincial life in Middlemarch with psychological depth.",
        # History — ancient & medieval
        "The Roman Republic gave way to the Empire under the rule of Augustus Caesar.",
        "The fall of Constantinople to the Ottoman Turks in fourteen fifty-three ended the Byzantine Empire.",
        "The Crusades brought the armies of Christendom into contact with the civilisations of the East.",
        "Alexander the Great conquered an empire stretching from Greece to the borders of India.",
        "The Athenian democracy allowed every free male citizen to participate directly in the government.",
        "The feudal system bound lord and vassal together in a hierarchy of obligation and service.",
        "The Black Death killed perhaps a third of the population of Europe in the fourteenth century.",
        "Charlemagne united much of Western Europe under a single crown and revived learning at his court.",
        "The Magna Carta established the principle that even the king is subject to the law.",
        "The Vikings raided the coasts of Europe and settled as far afield as Iceland and Normandy.",
        # History — early modern & modern
        "The French Revolution overthrew the monarchy and proclaimed the rights of man and the citizen.",
        "The American colonies declared their independence from Britain in seventeen seventy-six.",
        "The Congress of Vienna restored the balance of power in Europe after the defeat of Napoleon.",
        "The slave trade carried millions of Africans across the Atlantic to labour on plantations.",
        "The Reformation shattered the religious unity of Western Europe and led to prolonged wars of religion.",
        "The printing press made books affordable and spread literacy far beyond the monasteries and universities.",
        "The Spanish conquest of Mexico and Peru destroyed the Aztec and Inca empires.",
        "The Glorious Revolution of sixteen eighty-eight established parliamentary sovereignty in England.",
        "The Napoleonic Wars engulfed Europe for nearly two decades and redrawn the map of the continent.",
        "The unification of Germany under Prussian leadership transformed the balance of power in Europe.",
        # Law & jurisprudence
        "Blackstone systematised the common law of England in his Commentaries on the Laws of England.",
        "The presumption of innocence requires the prosecution to prove guilt beyond a reasonable doubt.",
        "The right of habeas corpus protects the individual against arbitrary detention by the state.",
        "Trial by jury places the determination of guilt or innocence in the hands of the accused's peers.",
        "Natural law theorists hold that unjust laws have no binding moral authority.",
        "The codification of law aimed to replace the confusion of local customs with a single rational system.",
        "International law governs the conduct of nations toward one another in peace and in war.",
        "The doctrine of precedent requires courts to follow the decisions of higher tribunals.",
        "The separation of church and state prevents the government from imposing a particular religion.",
        "The abolition of torture as a means of extracting confession marked a turning point in legal reform.",
        # Education & scholarship
        "The ancient universities of Oxford and Cambridge trained the clergy and the ruling class for centuries.",
        "The Enlightenment insisted that reason and evidence, not authority, should guide the pursuit of knowledge.",
        "The Royal Society was founded to promote the advancement of natural knowledge by experiment.",
        "The great encyclopaedias of the eighteenth century aimed to catalogue all human knowledge.",
        "Classical education centred on the study of Latin and Greek language and literature.",
        "The lyceum system brought secondary education within the reach of the middle classes.",
        "Public libraries opened their doors to readers who could not afford to buy their own books.",
        "The development of the research university in Germany set a new model for higher education.",
        "Philology traced the relationships among languages and reconstructed their common ancestors.",
        "The study of antiquities uncovered the remains of civilisations long buried beneath the soil.",
        # Extra humanities
        "Rhetoric was considered the queen of the arts and essential to public life in the ancient world.",
        "The humanists of the Renaissance revived the study of classical texts and celebrated human achievement.",
        "Historiography asks not only what happened in the past but how we can know it.",
        "The rights of man were debated with increasing urgency throughout the eighteenth century.",
        "The novel emerged as the dominant literary form of the nineteenth century.",
        "Epic poetry recounts the deeds of heroes and the founding of nations in elevated verse.",
        "Satire uses wit and irony to expose the follies and vices of society.",
        "The essay as a literary form was perfected by Montaigne and developed by Addison and Steele.",
        "Drama flourished in ancient Athens and was revived on the stages of Elizabethan England.",
        "The art of biography aims to capture the character and achievements of notable individuals.",
    ],
    "Daily Life & Technology": [
        # Agriculture — crops & methods
        "The four-course rotation of wheat, turnips, barley, and clover restored the fertility of the soil.",
        "The seed drill allowed farmers to plant in neat rows and greatly reduced the waste of seed.",
        "The enclosure of common lands concentrated farmland in the hands of larger and more efficient proprietors.",
        "The threshing machine separated the grain from the straw far more quickly than flailing by hand.",
        "Guano imported from South America was spread on the fields as a powerful natural fertiliser.",
        "Drainage of marshlands converted waterlogged ground into productive arable land.",
        "The cultivation of the potato provided a cheap and nutritious food for the labouring poor.",
        "Maize and tobacco were among the most important crops introduced from the New World.",
        "The sugar plantation depended upon the labour of enslaved Africans working under brutal conditions.",
        "The harvest was the most critical period of the agricultural year, demanding the labour of every hand.",
        "Cider was pressed from apples in the autumn and stored in barrels for the winter.",
        "The vineyard required careful attention to pruning, soil, and exposure to produce a fine vintage.",
        "Root vegetables such as turnips and carrots were grown as winter fodder for livestock.",
        "The practice of marling improved heavy clay soils by mixing them with calcium-rich earth.",
        "Crop failure brought famine, and the Irish potato blight caused a catastrophe of starvation and emigration.",
        # Agriculture — livestock
        "Selective breeding improved the size and quality of cattle, sheep, and horses over many generations.",
        "The merino sheep was prized for the fineness and weight of its fleece.",
        "Dairy farming supplied the cities with milk, butter, and cheese carried in by rail and cart.",
        "The horse was the principal source of motive power on the farm and on the road.",
        "Poultry were kept in every cottage yard and provided eggs and meat for the household.",
        "The ox was used for ploughing heavy soil before the horse gradually replaced it.",
        "Pig-keeping was common among the labouring poor, who fattened a hog on kitchen scraps.",
        "The farrier shod horses and treated their ailments with traditional remedies.",
        "The cattle fair brought buyers and sellers together and was a great occasion for the country town.",
        "Wool was shorn from the sheep in spring and sold to the merchants who supplied the textile mills.",
        # Industry & manufacturing
        "The spinning jenny multiplied the number of threads a single worker could spin at once.",
        "The power loom transformed weaving from a cottage craft into a factory operation.",
        "The blast furnace smelted iron ore with coke to produce pig iron on an industrial scale.",
        "The Bessemer process converted pig iron into steel cheaply and in large quantities.",
        "The factory system gathered workers under one roof and subjected them to the discipline of the clock.",
        "Child labour in the mills and mines was widespread before the passage of protective legislation.",
        "The steam hammer forged heavy pieces of iron and steel with great force and precision.",
        "Brickmaking supplied the material for the rapid expansion of towns and cities.",
        "The pottery industry produced earthenware and porcelain for domestic and export markets.",
        "Glassmaking required skilled workers to blow, cut, and polish the finished product.",
        "The tanning of hides produced leather for shoes, belts, harnesses, and bookbindings.",
        "Sawmills cut timber into planks for building houses, ships, and furniture.",
        "The gas works produced coal gas for lighting the streets and homes of the towns.",
        "The chemical industry manufactured acids, alkalis, dyes, and explosives in ever-growing quantities.",
        "The paper mill turned rags and later wood pulp into sheets of paper for printing and writing.",
        # Transport
        "The stage coach carried passengers and mail between the principal towns along the turnpike roads.",
        "The canal connected the coal fields with the factories and the ports with the interior.",
        "The railway revolutionised travel and transformed the economy by speeding the movement of goods.",
        "The steamship crossed the Atlantic in a fraction of the time required by a sailing vessel.",
        "The omnibus provided cheap public transport within the growing cities.",
        "The bicycle gave ordinary people a new freedom of movement on the open road.",
        "The suspension bridge carried roads and railways across rivers and gorges of unprecedented span.",
        "The tunnel bored through mountains and beneath rivers to shorten the route of the railway.",
        "The lighthouse warned ships of dangerous rocks and guided them safely into harbour.",
        "The packet boat carried mail, passengers, and light cargo on a regular schedule.",
        "The clipper ship was the fastest sailing vessel ever built and carried tea from China to London.",
        "The horse-drawn tram ran on iron rails laid in the surface of the city streets.",
        "The lock enabled canal boats to ascend and descend hills by raising and lowering the water level.",
        "The harbour master regulated the movement of ships and the loading and unloading of cargo.",
        "The covered wagon carried settlers and their possessions across the plains to the western territories.",
        # Communication
        "The electric telegraph transmitted messages over hundreds of miles in a matter of seconds.",
        "The penny post made it possible for anyone to send a letter anywhere in the country for a single stamp.",
        "The newspaper brought the events of the day to the breakfast table of the reading public.",
        "The rotary printing press produced thousands of copies per hour and made cheap newspapers possible.",
        "The typewriter began to replace the pen for commercial and official correspondence.",
        "The telephone transmitted the human voice along a wire and promised to transform communication.",
        "The postal service employed thousands of clerks, carriers, and coachmen to move the mail.",
        "The semaphore telegraph used mechanical arms on hilltop towers to signal messages across the country.",
        "The submarine cable carried telegraph messages beneath the Atlantic Ocean.",
        "The pigeon post carried dispatches during the siege when all other communications were cut.",
        # Household & domestic life
        "The candle and the oil lamp provided light in the home before the coming of gas and electricity.",
        "The kitchen range burned coal and served for cooking, baking, and heating water.",
        "The washday demanded hours of labour with soap, water, and a scrubbing board.",
        "The sewing machine greatly reduced the time required to make and mend clothing.",
        "The parlour was the best room in the house, reserved for receiving guests.",
        "Servants rose before dawn to light the fires and prepare the household for the day.",
        "The icebox preserved food in summer by means of blocks of ice delivered by the iceman.",
        "Linen was bleached by spreading it on the grass in the sunlight.",
        "The grandfather clock stood in the hall and marked the hours with a chiming bell.",
        "China tea sets and silverware were displayed in the dining room as marks of respectability.",
        "The cottage garden supplied vegetables, herbs, and flowers for the household.",
        "Spring cleaning involved beating carpets, washing curtains, and airing mattresses.",
        "The cast-iron stove replaced the open hearth and heated the room more efficiently.",
        "Matches replaced the tinderbox and made the lighting of fires quick and easy.",
        "The flat iron was heated on the stove and used to press shirts, sheets, and tablecloths.",
        # Commerce & finance
        "The merchant bought raw materials from abroad and sold manufactured goods in return.",
        "The joint-stock company allowed investors to pool their capital and share the risks of trade.",
        "The stock exchange provided a market where shares in companies could be bought and sold.",
        "The gold standard fixed the value of the currency to a definite weight of gold.",
        "The bill of exchange facilitated trade by allowing merchants to settle debts at a distance.",
        "Insurance protected ships and cargoes against the risks of loss at sea.",
        "The warehouse stored goods awaiting sale or shipment to their final destination.",
        "The auction sold property, livestock, and household goods to the highest bidder.",
        "The market day brought farmers and tradespeople together in the square of the country town.",
        "Banking houses advanced credit to governments and financed the expansion of industry and trade.",
        "The tariff protected domestic manufactures by imposing duties on imported goods.",
        "The free trade movement argued that the removal of tariffs would benefit producers and consumers alike.",
        "The depression brought unemployment, bank failures, and widespread hardship.",
        "Wages were often paid in truck rather than in coin, tying the worker to the employer's shop.",
        "The cooperative movement sought to give working people a share in the profits of the businesses they patronised.",
        # Crafts & trades
        "The blacksmith forged horseshoes, ploughshares, and tools at his anvil.",
        "The carpenter built houses, barns, and furniture from timber sawn to size.",
        "The mason laid bricks and dressed stone for the construction of buildings and walls.",
        "The tailor cut cloth and stitched garments to measure for his customers.",
        "The cobbler repaired boots and shoes and made new ones from leather.",
        "The baker kneaded dough and baked bread in a wood-fired or coal-fired oven.",
        "The brewer malted barley and brewed ale and beer for the public house and the home.",
        "The wheelwright built and repaired the wheels of carts, coaches, and wagons.",
        "The cooper made barrels and casks for storing ale, wine, and provisions.",
        "The printer set type by hand and pulled impressions on a flat-bed press.",
        # Medicine & health
        "The hospital was originally a charitable institution for the sick poor and only later a place of medical treatment.",
        "Antiseptic surgery dramatically reduced the rate of infection and death following operations.",
        "The stethoscope enabled the physician to listen to the sounds of the heart and lungs.",
        "Vaccination against smallpox was the first great triumph of preventive medicine.",
        "The miasma theory held that disease was caused by foul air arising from rotting matter.",
        "Cholera epidemics ravaged the crowded cities and spurred the movement for sanitary reform.",
        "The dispensary provided medicines and medical advice to the poor at little or no cost.",
        "Ether and chloroform were introduced as anaesthetics to relieve the pain of surgical operations.",
        "The midwife attended the mother during childbirth and cared for the newborn infant.",
        "Public health measures including clean water, sewerage, and food inspection saved countless lives.",
        # Extra daily life
        "The fair was a great occasion for entertainment, with jugglers, acrobats, and travelling players.",
        "The coaching inn provided food, drink, and a bed for travellers on the road.",
        "The village school taught reading, writing, and arithmetic to the children of the labouring poor.",
        "The workhouse sheltered the destitute but imposed harsh conditions and the stigma of pauperism.",
        "The almshouse provided accommodation for the aged and infirm poor of the parish.",
        "Sunday was a day of rest and church attendance observed by law and custom.",
        "The circulating library lent novels and other books to subscribers for a small fee.",
        "The tavern served as a meeting place where men gathered to drink, debate, and read the newspaper.",
        "The volunteer fire brigade turned out with hand-pumped engines to fight fires in the town.",
        "The lamplighter went through the streets at dusk lighting the gas lamps one by one.",
    ],
    "Geography & Culture": [
        # Europe
        "London grew into the largest city in the world, the capital of a vast commercial and colonial empire.",
        "Paris was the centre of fashion, learning, and revolution throughout the eighteenth and nineteenth centuries.",
        "Vienna served as the seat of the Habsburg emperors and a capital of music and diplomacy.",
        "Rome preserved the ruins of the ancient empire and served as the seat of the papacy.",
        "Berlin rose from a provincial garrison town to become the capital of the new German Empire.",
        "St Petersburg was built by Peter the Great as a window onto Europe and the seat of the Russian court.",
        "Madrid was the capital from which Spain governed its vast dominions in the Americas.",
        "Amsterdam prospered as a centre of trade, banking, and religious tolerance.",
        "The Swiss cantons maintained their independence and neutrality amid the wars of their neighbours.",
        "The Italian peninsula remained divided among rival states until unification in the nineteenth century.",
        "Athens was revered as the birthplace of democracy, philosophy, and the arts.",
        "Constantinople stood at the crossroads of Europe and Asia and controlled the straits.",
        "Edinburgh was renowned for its university and its contributions to philosophy and medicine.",
        "Dublin was the second city of the British Isles and the centre of Irish political and literary life.",
        "Lisbon commanded the mouth of the Tagus and served as the starting point for voyages of discovery.",
        # Americas
        "The Declaration of Independence proclaimed that all men are created equal and endowed with certain rights.",
        "The vast prairies of North America were home to immense herds of bison and to the native peoples.",
        "The gold rush drew thousands of prospectors to California in search of fortune.",
        "The Mississippi River served as the great highway of commerce for the interior of the continent.",
        "The plantation economy of the southern states depended upon the labour of enslaved Africans.",
        "The frontier moved steadily westward as settlers cleared the forests and broke the prairie sod.",
        "Brazil was the largest territory in South America and the last country in the hemisphere to abolish slavery.",
        "The Panama Canal was conceived as a passage between the Atlantic and the Pacific oceans.",
        "The Monroe Doctrine declared that the Americas were closed to further European colonisation.",
        "The Andes mountains run the entire length of South America from north to south.",
        "The Amazon river drains the largest tropical forest on earth.",
        "Mexico gained its independence from Spain early in the nineteenth century after a prolonged struggle.",
        "The Haitian revolution was the first successful slave revolt to establish an independent nation.",
        "The Canadian provinces were united in a confederation under the British Crown.",
        "The gaucho roamed the pampas of Argentina on horseback, herding cattle across the open grasslands.",
        # Asia & Middle East
        "The Mughal Empire ruled over much of the Indian subcontinent from its splendid capital at Delhi.",
        "China was governed by a vast imperial bureaucracy selected through competitive examination.",
        "Japan opened its ports to foreign trade after centuries of self-imposed isolation.",
        "The Silk Road carried goods and ideas between China and the Mediterranean for over a thousand years.",
        "The Ottoman Empire controlled the eastern Mediterranean and the overland routes to Asia.",
        "India was the jewel of the British Empire and a source of cotton, tea, and spices.",
        "Persia was celebrated for its poetry, its carpets, and the splendour of its royal courts.",
        "The Great Wall of China stretched for thousands of miles along the northern frontier.",
        "The opium trade provoked a war between Britain and China over the terms of commerce.",
        "The holy cities of Mecca and Medina drew pilgrims from every corner of the Islamic world.",
        "The samurai class upheld a code of honour and martial discipline in feudal Japan.",
        "The temples of Angkor in Cambodia were rediscovered overgrown by jungle in the nineteenth century.",
        "The Suez Canal shortened the voyage from Europe to Asia by thousands of miles.",
        "The tea trade with China was among the most lucrative branches of the East India commerce.",
        "Tibet was a remote and mountainous kingdom largely closed to outsiders.",
        # Africa & Oceania
        "The interior of Africa remained largely unknown to Europeans until the explorations of the nineteenth century.",
        "The slave trade depopulated vast regions of West Africa and enriched the merchants of the coast.",
        "Egypt fascinated European scholars with its pyramids, temples, and hieroglyphic inscriptions.",
        "The Zulu kingdom built a formidable military power in southern Africa.",
        "The Cape Colony at the southern tip of Africa was a vital way station on the route to India.",
        "The Sahara desert stretches across the entire breadth of northern Africa.",
        "Madagascar is the fourth largest island in the world and home to many species found nowhere else.",
        "The exploration of the Niger River opened West Africa to European commerce and colonisation.",
        "Australia was claimed by Britain and settled first as a penal colony for transported convicts.",
        "The Maori people of New Zealand developed a rich culture of carving, warfare, and oral tradition.",
        "The Pacific islands were charted by Cook and other navigators in the eighteenth century.",
        "The diamond and gold fields of southern Africa attracted fortune seekers from around the world.",
        "The tsetse fly made large areas of tropical Africa uninhabitable for horses and cattle.",
        "Livingstone followed the course of the Zambezi and reported on the peoples and landscapes he encountered.",
        "The partition of Africa among the European powers was agreed at the Berlin Conference.",
        # Empires & colonialism
        "The British Empire at its height governed a quarter of the world's population and land surface.",
        "The Spanish Empire once encompassed much of the Americas, the Philippines, and parts of Europe.",
        "The Portuguese established trading posts along the coasts of Africa, India, and Brazil.",
        "The Dutch East India Company controlled the spice trade and governed territories in Southeast Asia.",
        "The French established colonies in North America, the Caribbean, West Africa, and Southeast Asia.",
        "The Russian Empire expanded across Siberia to the Pacific and southward into Central Asia.",
        "The Ottoman Empire slowly contracted under pressure from nationalist movements and European encroachment.",
        "The Qing dynasty governed China but faced internal rebellion and external aggression in its final decades.",
        "The Austro-Hungarian Empire comprised a patchwork of peoples and languages held together by the Habsburg crown.",
        "Colonial rule imposed European laws, languages, and institutions on subject peoples around the world.",
        # Arts — visual arts
        "The Renaissance masters perfected the techniques of perspective, anatomy, and oil painting.",
        "Rembrandt captured the play of light and shadow in his portraits and biblical scenes.",
        "The Impressionists painted outdoors and sought to capture the fleeting effects of light and atmosphere.",
        "Turner painted the sea, the sky, and the forces of nature with a freedom that astonished his contemporaries.",
        "The engraver cut images into copper plates to produce prints for books and portfolios.",
        "Japanese woodblock prints influenced European artists with their bold outlines and flat areas of colour.",
        "The daguerreotype was the first practical method of producing a permanent photographic image.",
        "Porcelain from China and Japan was prized in Europe for its delicacy and beauty.",
        "The fresco adorned the walls and ceilings of churches and palaces with scenes from scripture and mythology.",
        "The Academy exhibition was the most important annual event in the artistic calendar.",
        # Arts — music & performance
        "The symphony orchestra grew in size and range throughout the eighteenth and nineteenth centuries.",
        "Opera combined music, drama, and spectacle on the stages of Milan, Vienna, and Paris.",
        "The piano became the most popular instrument for domestic music-making in the nineteenth century.",
        "Folk songs preserved the traditions and stories of the common people in every country.",
        "The waltz scandalised polite society when it was first danced in the ballrooms of Vienna.",
        "Church music ranged from the simplicity of the hymn to the grandeur of the oratorio.",
        "The travelling circus brought acrobats, clowns, and exotic animals to towns across the country.",
        "The music hall provided popular entertainment for the working classes in the cities.",
        "Choral societies brought together amateur singers to perform the great works of sacred and secular music.",
        "The organ filled the nave of the cathedral with sound and accompanied the singing of the congregation.",
        # Architecture & monuments
        "The Gothic cathedral soared heavenward with pointed arches, ribbed vaults, and walls of stained glass.",
        "The classical revival drew inspiration from the temples of Greece and Rome.",
        "The Crystal Palace was built of iron and glass to house the Great Exhibition of eighteen fifty-one.",
        "The Taj Mahal was built by the Mughal emperor as a tomb for his beloved wife.",
        "The Parthenon stands on the Acropolis as the supreme achievement of ancient Greek architecture.",
        "The great country houses of England were surrounded by landscaped parks and gardens.",
        "The mosque combined a place of prayer with a courtyard, a minaret, and often a school.",
        "The pagoda is a tower of multiple stories found in the temple complexes of China and Japan.",
        "The aqueduct carried water from distant sources to supply the cities of the Roman Empire.",
        "The railway station became a new type of public building, combining engineering with architectural display.",
        # Languages & customs
        "Comparative philology revealed the kinship of Sanskrit, Greek, Latin, and the Germanic tongues.",
        "The Rosetta Stone provided the key to deciphering the hieroglyphic script of ancient Egypt.",
        "The grammars and dictionaries compiled by missionaries preserved many languages that might otherwise have been lost.",
        "The custom of afternoon tea became a fixture of English social life in the nineteenth century.",
        "The carnival preceding Lent was celebrated with masks, costumes, and public festivity.",
        "The festival of the harvest gave thanks for the bounty of the fields and the safety of the crop.",
        "The wedding ceremony united families and was accompanied by feasting and celebration.",
        "The funeral procession conducted the dead to their resting place with solemnity and ritual.",
        "The art of calligraphy was cultivated in China, Japan, and the Islamic world as a high form of expression.",
        "The oral tradition preserved the myths, legends, and histories of peoples without a written language.",
    ],
}

for cat, sents in SENTENCES.items():
    print(f"  {cat}: {len(sents)} sentences")
total = sum(len(v) for v in SENTENCES.values())
print(f"  Total: {total} sentences")

# ---- Collect embeddings ----
print("\n=== Collecting sentence embeddings ===")
all_labels, all_categories, all_vecs = [], [], []

for cat, sents in SENTENCES.items():
    for s in sents:
        vec = get_sentence_embedding(s)
        # Use first few words as label
        all_labels.append(" ".join(s.split()[:5]))
        all_categories.append(cat)
        all_vecs.append(vec)
    print(f"  {cat}: done")

X = np.stack(all_vecs)
print(f"Embedding matrix: {X.shape}")

# Mean-center
X_centered = X - X.mean(axis=0, keepdims=True)
X_norm = X_centered / (np.linalg.norm(X_centered, axis=1, keepdims=True) + 1e-8)

# ---- UMAP ----
print("\n=== Running UMAP ===")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
umap_2d = reducer.fit_transform(X_norm)

# ---- PCA ----
pca = PCA(n_components=2)
pca_2d = pca.fit_transform(X_norm)
print(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")

# ---- Colors ----
cat_colors = {
    "Sciences": "#e74c3c",
    "Humanities": "#3498db",
    "Daily Life & Technology": "#2ecc71",
    "Geography & Culture": "#f39c12",
}

# ---- Plot 1: UMAP ----
fig, ax = plt.subplots(figsize=(16, 12))
for cat in cat_colors:
    idxs = [i for i, c in enumerate(all_categories) if c == cat]
    ax.scatter(umap_2d[idxs, 0], umap_2d[idxs, 1],
              c=cat_colors[cat], s=15, alpha=0.7, edgecolors='none', label=cat)
ax.legend(fontsize=11, markerscale=2.5, loc='best')
ax.set_title("UMAP of pre-1900 sentence embeddings (d34-22btok, final layer, 800 sentences)", fontsize=14)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, '14_umap_sentences.png'), bbox_inches='tight')
print("Saved 14_umap_sentences.png")
plt.close()

# ---- Plot 2: PCA ----
fig, ax = plt.subplots(figsize=(16, 12))
for cat in cat_colors:
    idxs = [i for i, c in enumerate(all_categories) if c == cat]
    ax.scatter(pca_2d[idxs, 0], pca_2d[idxs, 1],
              c=cat_colors[cat], s=15, alpha=0.7, edgecolors='none', label=cat)
ax.legend(fontsize=11, markerscale=2.5, loc='best')
ax.set_title("PCA of pre-1900 sentence embeddings (d34-22btok, final layer)", fontsize=14)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, '15_pca_sentences.png'), bbox_inches='tight')
print("Saved 15_pca_sentences.png")
plt.close()

# Cleanup
hook.remove()
print(f"\n=== Done! Figures saved to {FIGDIR} ===")
